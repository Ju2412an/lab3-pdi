# -*- coding: utf-8 -*-
"""
Vision Mamba (S6) desde cero — versión LOCAL (Windows + GTX 1060 6GB).

Adaptación del notebook de Colab para correr en local:
  - SIN Google Drive, SIN `!pip`: usa el dataset ya extraído en ./data
  - Config liviana para 6 GB de VRAM y subconjunto del dataset para que
    el entrenamiento termine en pocos minutos (modo "pre-correr resultados").
  - Guarda en ./results: curvas, matriz de confusión, predicciones,
    reporte de clasificación y el checkpoint del mejor modelo.

Uso:
    .venv\\Scripts\\python.exe train_local.py
"""
import os, time, json, math, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Configuración (ajustada a GTX 1060 6GB; subconjunto para demo rápida)
# ----------------------------------------------------------------------------
class CFG:
    data_root   = os.path.join('data', 'Plant Village Dataset')
    out_dir     = 'results'
    img_size    = 128      # 128/patch16 -> 8x8 = 64 tokens (scan más corto = más rápido)
    patch_size  = 16
    in_chans    = 3
    # --- modelo Vision Mamba ---
    embed_dim   = 160
    depth       = 4        # nº de bloques Mamba (menos que en Colab por velocidad)
    expand      = 2        # d_inner = expand * embed_dim
    d_state     = 16       # dimensión del estado del SSM (N)
    dt_rank     = 10       # rango de Delta (entrada-dependiente)
    conv_kernel = 4        # conv1d causal local
    bidirectional = True   # escaneo adelante + atrás (estilo Vim)
    drop        = 0.1
    # --- entrenamiento ---
    epochs      = 5
    batch_size  = 32       # 6 GB: 32 es seguro a 128px con AMP
    lr          = 4e-4
    weight_decay= 0.05
    label_smooth= 0.1
    num_workers = 4
    amp         = True     # mixed precision (fp16)
    seed        = 42
    # --- subconjunto (para que el pre-run termine rápido) ---
    train_per_class = 250  # imgs por clase en train (None = todas)
    val_per_class   = 60   # imgs por clase en val   (None = todas)
    # test se usa completo (1355 imgs)

cfg = CFG()


# ----------------------------------------------------------------------------
# Modelo: RMSNorm + MambaBlock (S6 selective scan) + VisionMamba
# (idéntico en espíritu al de la Parte A del notebook)
# ----------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))
    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class MambaBlock(nn.Module):
    """Bloque S6 (selective SSM) en PyTorch puro."""
    def __init__(self, cfg):
        super().__init__()
        self.d_model = cfg.embed_dim
        self.d_inner = cfg.expand * cfg.embed_dim
        self.n       = cfg.d_state
        self.dt_rank = cfg.dt_rank
        self.bidirectional = cfg.bidirectional

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner,
                                kernel_size=cfg.conv_kernel,
                                groups=self.d_inner,
                                padding=cfg.conv_kernel - 1, bias=True)
        self.x_proj  = nn.Linear(self.d_inner, self.dt_rank + 2 * self.n, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        A = torch.arange(1, self.n + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D     = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def selective_scan(self, u, delta, A, B, C, D):
        b, l, d_in = u.shape
        n = A.shape[1]
        h = u.new_zeros(b, d_in, n)
        ys = []
        for t in range(l):
            dt   = delta[:, t]
            dA   = torch.exp(dt.unsqueeze(-1) * A)
            dBu  = (dt.unsqueeze(-1) * B[:, t].unsqueeze(1)) * u[:, t].unsqueeze(-1)
            h    = dA * h + dBu
            y    = torch.einsum('bdn,bn->bd', h, C[:, t])
            ys.append(y)
        y = torch.stack(ys, dim=1)
        return y + u * D

    def ssm(self, x):
        A = -torch.exp(self.A_log.float())
        x_dbl = self.x_proj(x)
        delta, B, C = torch.split(x_dbl, [self.dt_rank, self.n, self.n], dim=-1)
        delta = F.softplus(self.dt_proj(delta))
        return self.selective_scan(x.float(), delta.float(), A,
                                   B.float(), C.float(), self.D.float())

    def forward(self, x):
        b, l, d = x.shape
        x_and_z = self.in_proj(x)
        xs, z = x_and_z.chunk(2, dim=-1)
        xs = xs.transpose(1, 2)
        xs = self.conv1d(xs)[..., :l]
        xs = xs.transpose(1, 2)
        xs = F.silu(xs)
        y = self.ssm(xs)
        if self.bidirectional:
            y = y + self.ssm(xs.flip(1)).flip(1)
        y = y * F.silu(z)
        return self.out_proj(y.to(x.dtype))


class ResidualMamba(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm = RMSNorm(cfg.embed_dim)
        self.mixer = MambaBlock(cfg)
    def forward(self, x):
        return x + self.mixer(self.norm(x))


class VisionMamba(nn.Module):
    def __init__(self, cfg, num_classes):
        super().__init__()
        self.patch = nn.Conv2d(cfg.in_chans, cfg.embed_dim,
                               kernel_size=cfg.patch_size, stride=cfg.patch_size)
        n_patches = (cfg.img_size // cfg.patch_size) ** 2
        self.pos = nn.Parameter(torch.zeros(1, n_patches, cfg.embed_dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.drop = nn.Dropout(cfg.drop)
        self.blocks = nn.ModuleList([ResidualMamba(cfg) for _ in range(cfg.depth)])
        self.norm_f = RMSNorm(cfg.embed_dim)
        self.head = nn.Linear(cfg.embed_dim, num_classes)

    def forward(self, x):
        x = self.patch(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.drop(x + self.pos)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm_f(x)
        x = x.mean(dim=1)
        return self.head(x)


# ----------------------------------------------------------------------------
def subset_indices(targets, per_class, num_classes, seed=42):
    """Índices estratificados: `per_class` por clase (None = todos)."""
    if per_class is None:
        return list(range(len(targets)))
    g = torch.Generator().manual_seed(seed)
    by_class = {c: [] for c in range(num_classes)}
    for i, t in enumerate(targets):
        by_class[t].append(i)
    out = []
    for c, idxs in by_class.items():
        idxs = [idxs[j] for j in torch.randperm(len(idxs), generator=g).tolist()]
        out.extend(idxs[:per_class])
    return out


def main():
    import torchvision.transforms as T
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader, Subset
    from torch.amp import autocast, GradScaler
    import matplotlib
    matplotlib.use('Agg')               # backend sin ventana (guardar a archivo)
    import matplotlib.pyplot as plt
    import numpy as np
    from collections import Counter
    from sklearn.metrics import classification_report, confusion_matrix

    os.makedirs(cfg.out_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(cfg.seed)
    print(f'device = {device}')
    if device == 'cuda':
        print('GPU:', torch.cuda.get_device_name(0))
    else:
        print('AVISO: no se detectó GPU NVIDIA. En CPU el selective scan es MUY lento.')
        print('       Para la demo, considera solo mostrar las figuras ya generadas en ./results/.')

    # --- el dataset no se versiona en GitHub (1 GB). Si falta, se extrae de archive.zip ---
    if not os.path.isdir(cfg.data_root):
        import zipfile
        if not os.path.exists('archive.zip'):
            raise SystemExit(
                "No se encontró el dataset.\n"
                f"  Falta la carpeta '{cfg.data_root}' y el archivo 'archive.zip'.\n"
                "  Coloca 'archive.zip' (PlantVillage) en la raíz del proyecto y vuelve a ejecutar.\n"
                "  (El dataset no se sube a GitHub por tamaño; pídelo a tu equipo.)")
        print("Descomprimiendo archive.zip -> ./data (1-3 min la primera vez)...")
        with zipfile.ZipFile('archive.zip') as z:
            z.extractall('data')
        print("Dataset listo.")

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    train_tf = T.Compose([
        T.RandomResizedCrop(cfg.img_size, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(15),
        T.ColorJitter(0.2, 0.2, 0.2),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = T.Compose([
        T.Resize(int(cfg.img_size * 1.14)),
        T.CenterCrop(cfg.img_size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_full = ImageFolder(os.path.join(cfg.data_root, 'Train'), transform=train_tf)
    val_full   = ImageFolder(os.path.join(cfg.data_root, 'Val'),   transform=eval_tf)
    test_ds    = ImageFolder(os.path.join(cfg.data_root, 'Test'),  transform=eval_tf)

    classes = train_full.classes
    num_classes = len(classes)

    train_ds = Subset(train_full, subset_indices(train_full.targets, cfg.train_per_class, num_classes, cfg.seed))
    val_ds   = Subset(val_full,   subset_indices(val_full.targets,   cfg.val_per_class,   num_classes, cfg.seed))

    print(f'Clases: {num_classes}')
    print(f'Train (subset)/Val (subset)/Test: {len(train_ds)} / {len(val_ds)} / {len(test_ds)}')

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
                          persistent_workers=cfg.num_workers > 0)
    val_dl   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                          num_workers=cfg.num_workers, pin_memory=True,
                          persistent_workers=cfg.num_workers > 0)
    test_dl  = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                          num_workers=cfg.num_workers, pin_memory=True,
                          persistent_workers=cfg.num_workers > 0)

    # --- balance de clases (train subset) ---
    sub_targets = [train_full.targets[i] for i in train_ds.indices]
    counts = Counter(sub_targets)
    plt.figure(figsize=(12, 4))
    plt.bar(range(num_classes), [counts[i] for i in range(num_classes)])
    plt.xticks(range(num_classes), classes, rotation=90, fontsize=7)
    plt.title('Imágenes por clase (Train subset)'); plt.tight_layout()
    plt.savefig(os.path.join(cfg.out_dir, 'clases_balance.png'), dpi=120); plt.close()

    # --- modelo ---
    model = VisionMamba(cfg, num_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Parámetros entrenables: {n_params/1e6:.2f} M')

    # sanity check del forward
    model.eval()
    with torch.no_grad():
        dummy = torch.randn(2, 3, cfg.img_size, cfg.img_size, device=device)
        out = model(dummy)
    assert out.shape == (2, num_classes), out.shape
    print('Sanity check forward OK:', tuple(out.shape))

    # --- topología (torchinfo) ---
    try:
        from torchinfo import summary
        s = summary(model, input_size=(1, 3, cfg.img_size, cfg.img_size),
                    col_names=['input_size', 'output_size', 'num_params'],
                    depth=3, verbose=0)
        with open(os.path.join(cfg.out_dir, 'topologia.txt'), 'w', encoding='utf-8') as f:
            f.write(str(s))
        print('Topología guardada en results/topologia.txt')
    except Exception as e:
        print('torchinfo no disponible:', e)

    # --- entrenamiento ---
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smooth)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs * len(train_dl))
    scaler = GradScaler('cuda', enabled=cfg.amp)

    @torch.no_grad()
    def evaluate(dl):
        model.eval()
        correct = total = 0; loss_sum = 0.0
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            with autocast('cuda', enabled=cfg.amp):
                o = model(x); loss = criterion(o, y)
            loss_sum += loss.item() * x.size(0)
            correct += (o.argmax(1) == y).sum().item()
            total += x.size(0)
        return loss_sum / total, correct / total

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_acc = 0.0
    ckpt_path = os.path.join(cfg.out_dir, 'vision_mamba_best.pt')

    for epoch in range(cfg.epochs):
        model.train(); t0 = time.time()
        run_loss = run_correct = run_total = 0
        for it, (x, y) in enumerate(train_dl):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda', enabled=cfg.amp):
                o = model(x); loss = criterion(o, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update(); scheduler.step()
            run_loss += loss.item() * x.size(0)
            run_correct += (o.argmax(1) == y).sum().item()
            run_total += x.size(0)
            if it % 50 == 0:
                print(f'  ep{epoch+1} it{it}/{len(train_dl)} '
                      f'loss {run_loss/run_total:.3f} acc {run_correct/run_total:.3f}')
        tr_loss, tr_acc = run_loss/run_total, run_correct/run_total
        va_loss, va_acc = evaluate(val_dl)
        history['train_loss'].append(tr_loss); history['train_acc'].append(tr_acc)
        history['val_loss'].append(va_loss);   history['val_acc'].append(va_acc)
        print(f'[Época {epoch+1}/{cfg.epochs}] train_loss {tr_loss:.3f} acc {tr_acc:.3f} | '
              f'val_loss {va_loss:.3f} acc {va_acc:.3f} | {time.time()-t0:.0f}s')
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({'model': model.state_dict(), 'classes': classes,
                        'cfg': {k: v for k, v in vars(CFG).items() if not k.startswith('__')}},
                       ckpt_path)
            print(f'  >> mejor modelo guardado (val_acc {best_acc:.3f})')

    print('\nMejor val_acc:', round(best_acc, 4))

    # --- curvas ---
    ep = range(1, len(history['train_loss'])+1)
    plt.figure(figsize=(11, 4))
    plt.subplot(1, 2, 1)
    plt.plot(ep, history['train_loss'], '-o', label='train')
    plt.plot(ep, history['val_loss'], '-o', label='val')
    plt.title('Loss'); plt.xlabel('época'); plt.legend(); plt.grid(alpha=.3)
    plt.subplot(1, 2, 2)
    plt.plot(ep, history['train_acc'], '-o', label='train')
    plt.plot(ep, history['val_acc'], '-o', label='val')
    plt.title('Accuracy'); plt.xlabel('época'); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(cfg.out_dir, 'curvas.png'), dpi=120); plt.close()

    # --- evaluación en test con el mejor checkpoint ---
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    test_loss, test_acc = evaluate(test_dl)
    print(f'TEST  loss {test_loss:.3f}  |  accuracy {test_acc:.4f}')

    model.eval(); all_pred, all_true = [], []
    with torch.no_grad():
        for x, y in test_dl:
            x = x.to(device)
            with autocast('cuda', enabled=cfg.amp):
                o = model(x)
            all_pred.extend(o.argmax(1).cpu().tolist())
            all_true.extend(y.tolist())

    report = classification_report(all_true, all_pred, target_names=classes, digits=3)
    print(report)
    with open(os.path.join(cfg.out_dir, 'classification_report.txt'), 'w', encoding='utf-8') as f:
        f.write(f'TEST accuracy: {test_acc:.4f}\n\n{report}')

    # --- matriz de confusión ---
    cm = confusion_matrix(all_true, all_pred)
    plt.figure(figsize=(11, 9))
    plt.imshow(cm, cmap='Blues'); plt.colorbar(fraction=0.046)
    plt.xticks(range(num_classes), classes, rotation=90, fontsize=6)
    plt.yticks(range(num_classes), classes, fontsize=6)
    plt.xlabel('Predicho'); plt.ylabel('Real')
    plt.title(f'Matriz de confusión — Test (acc {test_acc:.3f})')
    plt.tight_layout(); plt.savefig(os.path.join(cfg.out_dir, 'matriz_confusion.png'), dpi=120); plt.close()

    # --- predicciones de ejemplo ---
    def denorm(t):
        t = t.clone()
        for c in range(3):
            t[c] = t[c]*IMAGENET_STD[c] + IMAGENET_MEAN[c]
        return t.clamp(0, 1).permute(1, 2, 0).numpy()

    # muestra aleatoria (variada) en vez del primer lote, que sería de una sola clase
    g = torch.Generator().manual_seed(7)
    sel = torch.randperm(len(test_ds), generator=g)[:12].tolist()
    imgs = torch.stack([test_ds[i][0] for i in sel])
    labels = torch.tensor([test_ds[i][1] for i in sel])
    with torch.no_grad():
        with autocast('cuda', enabled=cfg.amp):
            preds = model(imgs.to(device)).argmax(1).cpu()
    plt.figure(figsize=(13, 8))
    for i in range(min(12, imgs.size(0))):
        plt.subplot(3, 4, i+1); plt.imshow(denorm(imgs[i])); plt.axis('off')
        ok = preds[i] == labels[i]
        plt.title(f'real: {classes[labels[i]]}\npred: {classes[preds[i]]}',
                  color='green' if ok else 'red', fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(cfg.out_dir, 'predicciones.png'), dpi=120); plt.close()

    # --- resumen JSON ---
    with open(os.path.join(cfg.out_dir, 'resumen.json'), 'w', encoding='utf-8') as f:
        json.dump({'test_acc': test_acc, 'best_val_acc': best_acc,
                   'history': history, 'num_classes': num_classes,
                   'params_M': n_params/1e6}, f, indent=2, ensure_ascii=False)

    print('\nListo. Resultados guardados en ./results/:')
    for fn in sorted(os.listdir(cfg.out_dir)):
        print('  -', fn)


if __name__ == '__main__':
    main()
