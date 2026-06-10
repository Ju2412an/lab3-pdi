# -*- coding: utf-8 -*-
"""Genera Vision_Mamba_PlantVillage_LOCAL.ipynb — versión para correr en LOCAL
(Windows + GPU NVIDIA, sin Google Drive ni `!pip` de Colab)."""
import json

cells = []
def md(t):   cells.append({"cell_type": "markdown", "metadata": {}, "source": t})
def code(t): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t})

# ----------------------------------------------------------------------------
md(r"""# Mamba (SSM) para visión — Clasificación de enfermedades en plantas 🌿  ·  **versión LOCAL**

**Tarea 3 · Punto 2 — Procesamiento de Imágenes · Equipo 2**
**Arquitectura: Mamba (2023) — modelo SSM eficiente · Dataset: PlantVillage (29 clases)**

---

### Diferencias con la versión de Colab
- **Sin Google Drive ni `!pip`:** el dataset ya está extraído en `./data/Plant Village Dataset` y las librerías ya están instaladas en el entorno local (`.venv`, Python 3.12, PyTorch CUDA).
- **Config ajustada a la GPU local (GTX 1060, 6 GB):** imágenes a 128 px, 4 bloques Mamba y un **subconjunto** del dataset (250 img/clase) para que el entrenamiento termine en ~25 min. El cuello de botella es el *selective scan* en PyTorch puro (secuencial), no la memoria.
- **Parte B (MambaVision / transfer learning):** requiere compilar los kernels CUDA de `mamba-ssm`/`causal-conv1d`, algo poco fiable en Windows + GPU Pascal. Se deja documentada y opcional al final.

### Mamba en una frase
A diferencia del Transformer (atención $O(L^2)$), Mamba procesa la secuencia con una **recurrencia de espacio de estados** $O(L)$, donde los parámetros del SSM ($\Delta, B, C$) **dependen de la entrada** (mecanismo *selectivo*): el modelo decide qué recordar y qué olvidar en cada paso.

$$ h_t = \bar{A}\, h_{t-1} + \bar{B}\, x_t \qquad y_t = C\, h_t + D\, x_t $$
""")

# ----------------------------------------------------------------------------
md(r"""## 0. GPU disponible""")
code("""import torch
print('PyTorch:', torch.__version__)
print('CUDA disponible:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('VRAM (GB):', round(torch.cuda.get_device_properties(0).total_memory/1e9, 1))""")

# ----------------------------------------------------------------------------
md(r"""## 1. Datos — dataset local

El dataset PlantVillage ya viene extraído en `./data/Plant Village Dataset`. Si solo tienes `archive.zip`, descomenta la celda de extracción.

```
data/Plant Village Dataset/
├── Train/  (53.691 img · 29 clases)
├── Val/    (12.067 img)
└── Test/   ( 1.355 img)
```""")
code("""import os
DATA_ROOT = os.path.join('data', 'Plant Village Dataset')

# --- (opcional) si solo tienes archive.zip, descomenta para extraer ---
# import zipfile
# if not os.path.isdir(DATA_ROOT):
#     with zipfile.ZipFile('archive.zip') as z:
#         z.extractall('data')

for split in ['Train', 'Val', 'Test']:
    p = os.path.join(DATA_ROOT, split)
    print(split, '->', os.path.exists(p), '|', len(os.listdir(p)) if os.path.exists(p) else 0, 'clases')""")

# ----------------------------------------------------------------------------
md(r"""## 2. Configuración (hiperparámetros)

Pensados para la **GTX 1060 (6 GB)**. El subconjunto (`train_per_class`) hace que el pre-entrenamiento sea rápido para la demo; pon `None` para usar el dataset completo.""")
code("""class CFG:
    data_root   = DATA_ROOT
    img_size    = 128      # 128/patch16 -> 8x8 = 64 tokens
    patch_size  = 16
    in_chans    = 3
    # --- modelo Vision Mamba ---
    embed_dim   = 160
    depth       = 4        # nº de bloques Mamba
    expand      = 2        # d_inner = expand * embed_dim
    d_state     = 16       # dimensión del estado del SSM (N)
    dt_rank     = 10       # rango de Delta (entrada-dependiente)
    conv_kernel = 4        # conv1d causal local
    bidirectional = True   # escaneo adelante + atrás (estilo Vim)
    drop        = 0.1
    # --- entrenamiento ---
    epochs      = 5
    batch_size  = 32
    lr          = 4e-4
    weight_decay= 0.05
    label_smooth= 0.1
    num_workers = 4        # en Windows, requiere ejecutar dentro de `if __name__`
    amp         = True     # mixed precision (fp16)
    seed        = 42
    # --- subconjunto para demo (None = dataset completo) ---
    train_per_class = 250
    val_per_class   = 60

cfg = CFG()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(cfg.seed)
print('device =', device)""")

# ----------------------------------------------------------------------------
md(r"""## 3. Datasets y DataLoaders

`ImageFolder` toma cada subcarpeta como una clase. Aumentos suaves en *train*; *val/test* sin aumentos. Para la demo tomamos un subconjunto estratificado (`per_class`) por velocidad.""")
code("""import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_tf = T.Compose([
    T.RandomResizedCrop(cfg.img_size, scale=(0.7, 1.0)),
    T.RandomHorizontalFlip(), T.RandomRotation(15),
    T.ColorJitter(0.2, 0.2, 0.2),
    T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
eval_tf = T.Compose([
    T.Resize(int(cfg.img_size * 1.14)), T.CenterCrop(cfg.img_size),
    T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

def subset_indices(targets, per_class, num_classes, seed=42):
    if per_class is None: return list(range(len(targets)))
    g = torch.Generator().manual_seed(seed)
    by_class = {c: [] for c in range(num_classes)}
    for i, t in enumerate(targets): by_class[t].append(i)
    out = []
    for c, idxs in by_class.items():
        idxs = [idxs[j] for j in torch.randperm(len(idxs), generator=g).tolist()]
        out.extend(idxs[:per_class])
    return out

train_full = ImageFolder(os.path.join(cfg.data_root, 'Train'), transform=train_tf)
val_full   = ImageFolder(os.path.join(cfg.data_root, 'Val'),   transform=eval_tf)
test_ds    = ImageFolder(os.path.join(cfg.data_root, 'Test'),  transform=eval_tf)

classes = train_full.classes
num_classes = len(classes)
train_ds = Subset(train_full, subset_indices(train_full.targets, cfg.train_per_class, num_classes, cfg.seed))
val_ds   = Subset(val_full,   subset_indices(val_full.targets,   cfg.val_per_class,   num_classes, cfg.seed))
print('Clases:', num_classes)
print('Train/Val/Test:', len(train_ds), len(val_ds), len(test_ds))

# En notebook (Windows) usa num_workers=0 para evitar problemas de multiprocessing;
# si lo ejecutas como script .py con `if __name__=='__main__'`, puedes subirlo.
NW = 0
train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  num_workers=NW, pin_memory=True, drop_last=True)
val_dl   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, num_workers=NW, pin_memory=True)
test_dl  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False, num_workers=NW, pin_memory=True)""")

md(r"""### 3.1 Vistazo a las imágenes y balance de clases""")
code("""import matplotlib.pyplot as plt
from collections import Counter

sub_targets = [train_full.targets[i] for i in train_ds.indices]
counts = Counter(sub_targets)
plt.figure(figsize=(12,4))
plt.bar(range(num_classes), [counts[i] for i in range(num_classes)])
plt.xticks(range(num_classes), classes, rotation=90, fontsize=7)
plt.title('Imágenes por clase (Train subset)'); plt.tight_layout(); plt.show()

def denorm(t):
    t = t.clone()
    for c in range(3): t[c] = t[c]*IMAGENET_STD[c] + IMAGENET_MEAN[c]
    return t.clamp(0,1).permute(1,2,0).numpy()

imgs, labels = next(iter(train_dl))
plt.figure(figsize=(12,6))
for i in range(12):
    plt.subplot(3,4,i+1); plt.imshow(denorm(imgs[i])); plt.axis('off')
    plt.title(classes[labels[i]], fontsize=7)
plt.tight_layout(); plt.show()""")

# ----------------------------------------------------------------------------
md(r"""# PARTE A — Vision Mamba desde cero

## 4. La arquitectura Vision Mamba — capa por capa

```
Imagen (3,128,128) → PatchEmbed (Conv2d 16x16) → 64 tokens de dim 160 (+ pos)
  → N× [ RMSNorm → MambaBlock(S6) → residual ]
  → RMSNorm → Mean Pool → Linear(160 → 29)
```

**MambaBlock (S6):** `in_proj → (x, z)` · `Conv1d causal + SiLU` (contexto local) ·
SSM selectivo donde $\Delta, B, C = f(x)$ y $A = -\exp(A\_log)$ · *selective scan* (recurrencia $O(L)$) ·
compuerta `y · SiLU(z)` · `out_proj`. **Bidireccional** porque una imagen no tiene orden causal.""")
code(r'''import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__(); self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class MambaBlock(nn.Module):
    """Bloque S6 (selective SSM) en PyTorch puro."""
    def __init__(self, cfg):
        super().__init__()
        self.d_model = cfg.embed_dim; self.d_inner = cfg.expand * cfg.embed_dim
        self.n = cfg.d_state; self.dt_rank = cfg.dt_rank; self.bidirectional = cfg.bidirectional
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=cfg.conv_kernel,
                                groups=self.d_inner, padding=cfg.conv_kernel - 1, bias=True)
        self.x_proj  = nn.Linear(self.d_inner, self.dt_rank + 2*self.n, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        A = torch.arange(1, self.n + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A)); self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def selective_scan(self, u, delta, A, B, C, D):
        b, l, d_in = u.shape; n = A.shape[1]
        h = u.new_zeros(b, d_in, n); ys = []
        for t in range(l):
            dt = delta[:, t]
            dA = torch.exp(dt.unsqueeze(-1) * A)
            dBu = (dt.unsqueeze(-1) * B[:, t].unsqueeze(1)) * u[:, t].unsqueeze(-1)
            h = dA * h + dBu
            ys.append(torch.einsum('bdn,bn->bd', h, C[:, t]))
        return torch.stack(ys, dim=1) + u * D

    def ssm(self, x):
        A = -torch.exp(self.A_log.float())
        delta, B, C = torch.split(self.x_proj(x), [self.dt_rank, self.n, self.n], dim=-1)
        delta = F.softplus(self.dt_proj(delta))
        return self.selective_scan(x.float(), delta.float(), A, B.float(), C.float(), self.D.float())

    def forward(self, x):
        b, l, d = x.shape
        xs, z = self.in_proj(x).chunk(2, dim=-1)
        xs = self.conv1d(xs.transpose(1, 2))[..., :l].transpose(1, 2)
        xs = F.silu(xs)
        y = self.ssm(xs)
        if self.bidirectional: y = y + self.ssm(xs.flip(1)).flip(1)
        y = y * F.silu(z)
        return self.out_proj(y.to(x.dtype))

class ResidualMamba(nn.Module):
    def __init__(self, cfg):
        super().__init__(); self.norm = RMSNorm(cfg.embed_dim); self.mixer = MambaBlock(cfg)
    def forward(self, x): return x + self.mixer(self.norm(x))

class VisionMamba(nn.Module):
    def __init__(self, cfg, num_classes):
        super().__init__()
        self.patch = nn.Conv2d(cfg.in_chans, cfg.embed_dim, cfg.patch_size, cfg.patch_size)
        n_patches = (cfg.img_size // cfg.patch_size) ** 2
        self.pos = nn.Parameter(torch.zeros(1, n_patches, cfg.embed_dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.drop = nn.Dropout(cfg.drop)
        self.blocks = nn.ModuleList([ResidualMamba(cfg) for _ in range(cfg.depth)])
        self.norm_f = RMSNorm(cfg.embed_dim); self.head = nn.Linear(cfg.embed_dim, num_classes)
    def forward(self, x):
        x = self.patch(x).flatten(2).transpose(1, 2)
        x = self.drop(x + self.pos)
        for blk in self.blocks: x = blk(x)
        return self.head(self.norm_f(x).mean(dim=1))

model = VisionMamba(cfg, num_classes).to(device)
print(f'Parámetros: {sum(p.numel() for p in model.parameters())/1e6:.2f} M')''')

md(r"""### 4.1 Sanity check del forward + topología""")
code("""model.eval()
with torch.no_grad():
    out = model(torch.randn(2, 3, cfg.img_size, cfg.img_size, device=device))
assert out.shape == (2, num_classes)
print('forward OK:', tuple(out.shape))

from torchinfo import summary
summary(model, input_size=(1, 3, cfg.img_size, cfg.img_size),
        col_names=['input_size', 'output_size', 'num_params'], depth=3)""")

# ----------------------------------------------------------------------------
md(r"""## 5. Entrenamiento
AdamW + cosine schedule, label smoothing 0.1, AMP (el *scan* se fuerza a fp32).""")
code("""from torch.amp import autocast, GradScaler
import time

criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smooth)
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs * len(train_dl))
scaler = GradScaler('cuda', enabled=cfg.amp)

@torch.no_grad()
def evaluate(dl):
    model.eval(); correct = total = 0; loss_sum = 0.0
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        with autocast('cuda', enabled=cfg.amp):
            o = model(x); loss = criterion(o, y)
        loss_sum += loss.item()*x.size(0); correct += (o.argmax(1)==y).sum().item(); total += x.size(0)
    return loss_sum/total, correct/total

history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
best_acc = 0.0
for epoch in range(cfg.epochs):
    model.train(); t0 = time.time(); run_loss = run_c = run_t = 0
    for it, (x, y) in enumerate(train_dl):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast('cuda', enabled=cfg.amp):
            o = model(x); loss = criterion(o, y)
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); scheduler.step()
        run_loss += loss.item()*x.size(0); run_c += (o.argmax(1)==y).sum().item(); run_t += x.size(0)
        if it % 50 == 0:
            print(f'  ep{epoch+1} it{it}/{len(train_dl)} loss {run_loss/run_t:.3f} acc {run_c/run_t:.3f}')
    tr_loss, tr_acc = run_loss/run_t, run_c/run_t
    va_loss, va_acc = evaluate(val_dl)
    for k, v in zip(history, [tr_loss, tr_acc, va_loss, va_acc]): history[k].append(v)
    print(f'[Época {epoch+1}/{cfg.epochs}] train {tr_loss:.3f}/{tr_acc:.3f} | val {va_loss:.3f}/{va_acc:.3f} | {time.time()-t0:.0f}s')
    if va_acc > best_acc:
        best_acc = va_acc
        torch.save({'model': model.state_dict(), 'classes': classes}, 'vision_mamba_best.pt')
print('\\nMejor val_acc:', round(best_acc, 4))""")

md(r"""### 5.1 Curvas de entrenamiento""")
code("""ep = range(1, len(history['train_loss'])+1)
plt.figure(figsize=(11,4))
plt.subplot(1,2,1); plt.plot(ep, history['train_loss'],'-o',label='train'); plt.plot(ep, history['val_loss'],'-o',label='val')
plt.title('Loss'); plt.xlabel('época'); plt.legend(); plt.grid(alpha=.3)
plt.subplot(1,2,2); plt.plot(ep, history['train_acc'],'-o',label='train'); plt.plot(ep, history['val_acc'],'-o',label='val')
plt.title('Accuracy'); plt.xlabel('época'); plt.legend(); plt.grid(alpha=.3)
plt.tight_layout(); plt.show()""")

# ----------------------------------------------------------------------------
md(r"""## 6. Evaluación en Test""")
code("""ckpt = torch.load('vision_mamba_best.pt', map_location=device)
model.load_state_dict(ckpt['model'])
test_loss, test_acc = evaluate(test_dl)
acc_scratch = test_acc
print(f'TEST  loss {test_loss:.3f}  |  accuracy {test_acc:.4f}')""")
code("""from sklearn.metrics import classification_report, confusion_matrix
model.eval(); all_pred, all_true = [], []
with torch.no_grad():
    for x, y in test_dl:
        with autocast('cuda', enabled=cfg.amp):
            o = model(x.to(device))
        all_pred.extend(o.argmax(1).cpu().tolist()); all_true.extend(y.tolist())
print(classification_report(all_true, all_pred, target_names=classes, digits=3))""")

md(r"""### 6.1 Matriz de confusión""")
code("""cm = confusion_matrix(all_true, all_pred)
plt.figure(figsize=(11,9)); plt.imshow(cm, cmap='Blues'); plt.colorbar(fraction=0.046)
plt.xticks(range(num_classes), classes, rotation=90, fontsize=6)
plt.yticks(range(num_classes), classes, fontsize=6)
plt.xlabel('Predicho'); plt.ylabel('Real')
plt.title(f'Matriz de confusión — Test (acc {test_acc:.3f})'); plt.tight_layout(); plt.show()""")

md(r"""### 6.2 Predicciones de ejemplo""")
code("""imgs, labels = next(iter(test_dl))
with torch.no_grad():
    with autocast('cuda', enabled=cfg.amp):
        preds = model(imgs.to(device)).argmax(1).cpu()
plt.figure(figsize=(13,8))
for i in range(12):
    plt.subplot(3,4,i+1); plt.imshow(denorm(imgs[i])); plt.axis('off')
    ok = preds[i] == labels[i]
    plt.title(f'real: {classes[labels[i]]}\\npred: {classes[preds[i]]}', color='green' if ok else 'red', fontsize=7)
plt.tight_layout(); plt.show()""")

# ----------------------------------------------------------------------------
md(r"""# PARTE B — Transfer Learning (opcional, requiere kernels CUDA)

> El profesor pidió un *re-entrenamiento*: tomar un Mamba preentrenado, quitarle la cabeza de 1000 clases (ImageNet) y poner una de **29 clases** (PlantVillage).

**MambaVision (NVIDIA)** necesita compilar `mamba-ssm` y `causal-conv1d` (kernels CUDA). En **Windows + GPU Pascal (GTX 1060)** esa compilación suele fallar, por eso esta parte es **opcional**. Si tienes Linux/WSL2 con una GPU más nueva, las celdas siguientes funcionan; en caso contrario, esta sección se explica con los resultados de Colab.""")
code("""# Ejecuta solo si tu entorno puede compilar mamba-ssm (Linux/WSL2 recomendado).
# !pip install mambavision causal-conv1d mamba-ssm transformers timm
try:
    from transformers import AutoModelForImageClassification
    mv = AutoModelForImageClassification.from_pretrained('nvidia/MambaVision-T-1K', trust_remote_code=True)
    in_features = mv.model.head.in_features
    mv.model.head = nn.Linear(in_features, num_classes)
    mv = mv.to(device)
    print('MambaVision cargado. Cabeza nueva:', mv.model.head)
except Exception as e:
    print('Parte B no disponible en este entorno (esperado en Windows):')
    print(type(e).__name__, str(e)[:300])""")

md(r"""## 7. Conclusiones

- **Parte A:** implementamos **Vision Mamba desde cero** (bloque S6 con *selective scan*) y entrenamos localmente en la GTX 1060 → demuestra que entendemos **cada capa** (Punto 1).
- Mamba reemplaza la **atención $O(L^2)$** del Transformer por una **recurrencia SSM $O(L)$**: escala linealmente con la longitud de la secuencia.
- **Limitación de la demo:** el *scan* en PyTorch puro es ~10-20× más lento que los kernels CUDA de `mamba-ssm`; por eso usamos un subconjunto y un modelo pequeño. La accuracy sube con más datos/épocas/profundidad.

| Componente | Función |
|---|---|
| PatchEmbed | Imagen → secuencia de tokens |
| in_proj → (x, z) | Expansión + rama de compuerta |
| Conv1d causal | Contexto **local** entre tokens |
| SSM selectivo (Δ,B,C = f(x)) | Filtrado **dinámico** dependiente de la entrada |
| A = −exp(A_log) | Matriz de estado **estable** |
| Selective scan | Recurrencia global en **tiempo lineal** |
| y · SiLU(z) | Compuerta multiplicativa |
| Bidireccional | Adapta el SSM (causal) a imágenes (sin orden) |
""")

# ----------------------------------------------------------------------------
for c in cells:
    c["source"] = c["source"].splitlines(keepends=True)

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3 (.venv)", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}

out = "Vision_Mamba_PlantVillage_LOCAL.ipynb"
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("Notebook escrito:", out, "| celdas:", len(cells))
