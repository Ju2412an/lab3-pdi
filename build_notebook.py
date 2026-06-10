# -*- coding: utf-8 -*-
"""Generador del notebook Vision Mamba (PlantVillage) para Google Colab."""
import json

cells = []

def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text})

def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": text})

# ----------------------------------------------------------------------------
md(r"""# Mamba (SSM) para visión — Clasificación de enfermedades en plantas 🌿

**Tarea 3 · Punto 2 (60%) — Procesamiento de Imágenes**
**Equipo 2 · Arquitectura seleccionada: Mamba (2023) — modelo SSM eficiente (Visión y lenguaje)**
**Exposición: 8 y 10 de junio · Dataset: PlantVillage (29 clases)**

---

### ¿Qué hace este notebook? (dos partes)
- **Parte A — Vision Mamba desde cero (PyTorch puro):** implementamos a mano el bloque **S6** (SSM selectivo + *selective scan*) y entrenamos. Sirve para **entender y explicar cada capa** (apoyo al Punto 1).
- **Parte B — Transfer Learning con un Mamba preentrenado (MambaVision, NVIDIA):** tomamos un modelo Mamba ya entrenado en ImageNet, le **cambiamos la cabeza** y hacemos **fine-tuning** en PlantVillage. Esto es lo que el profesor pidió explícitamente (*"re-entrenamiento / Transfer Learning: se le quitan las entradas y salidas y se ponen las de uno"*) y suele dar **mejor accuracy**.

> Así cubrimos lo pedagógico (Parte A, explicar cada capa) **y** el transfer learning (Parte B).

### Mamba en una frase
A diferencia del Transformer (atención $O(L^2)$), Mamba procesa la secuencia con una **recurrencia de espacio de estados** $O(L)$, donde los parámetros del SSM ($\Delta, B, C$) **dependen de la entrada** (mecanismo *selectivo*): el modelo decide qué recordar y qué olvidar en cada paso.

$$ h_t = \bar{A}\, h_{t-1} + \bar{B}\, x_t \qquad y_t = C\, h_t + D\, x_t $$
""")

md(r"""## 0. GPU disponible
Colab → Menú **Entorno de ejecución → Cambiar tipo de entorno → T4 GPU**.""")

code("""import torch
print('PyTorch:', torch.__version__)
print('CUDA disponible:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('VRAM (GB):', round(torch.cuda.get_device_properties(0).total_memory/1e9, 1))""")

# ----------------------------------------------------------------------------
md(r"""## 1. Datos — subir y descomprimir el dataset

**Pasos (una sola vez):**
1. Sube `archive.zip` (el PlantVillage que te dieron) a tu **Google Drive** (raíz de *Mi unidad*).
2. Ejecuta la celda → se monta Drive, se copia el zip a disco local de Colab (`/content`) y se descomprime ahí (lectura mucho más rápida que desde Drive).

Estructura esperada tras descomprimir:
```
Plant Village Dataset/
├── Train/  (53.693 img · 29 clases)
├── Val/    (12.067 img)
└── Test/   ( 1.358 img)
```""")

code("""import os, shutil, time

DRIVE_ZIP = '/content/drive/MyDrive/archive.zip'   # <-- ajusta si lo subiste a otra carpeta
LOCAL_ZIP = '/content/archive.zip'
DATA_ROOT = '/content/Plant Village Dataset'

from google.colab import drive
drive.mount('/content/drive')

if not os.path.exists(DATA_ROOT):
    if not os.path.exists(LOCAL_ZIP):
        print('Copiando zip a disco local...')
        shutil.copy(DRIVE_ZIP, LOCAL_ZIP)
    print('Descomprimiendo (puede tardar 1-3 min)...')
    t0 = time.time()
    import zipfile
    with zipfile.ZipFile(LOCAL_ZIP) as z:
        z.extractall('/content')
    print(f'Listo en {time.time()-t0:.0f}s')
else:
    print('Dataset ya descomprimido.')

for split in ['Train', 'Val', 'Test']:
    p = os.path.join(DATA_ROOT, split)
    print(split, '->', os.path.exists(p))""")

# ----------------------------------------------------------------------------
md(r"""## 2. Configuración (hiperparámetros)

Pensados para una **T4 (16 GB)**. Si te quedas sin memoria (OOM), baja `BATCH_SIZE` o `IMG_SIZE`.
El *selective scan* en PyTorch puro es ~10-20× más lento que los kernels CUDA: con el dataset completo cuenta **~10-20 min por época**. 3-5 épocas bastan para una buena demo.""")

code("""class CFG:
    data_root   = DATA_ROOT
    img_size    = 160      # 160/patch16 -> 10x10 = 100 tokens
    patch_size  = 16
    in_chans    = 3
    # --- modelo Vision Mamba ---
    embed_dim   = 192
    depth       = 6        # nº de bloques Mamba
    expand      = 2        # d_inner = expand * embed_dim
    d_state     = 16       # dimensión del estado del SSM (N)
    dt_rank     = 12       # rango de Delta (entrada-dependiente)
    conv_kernel = 4        # conv1d causal local
    bidirectional = True   # escaneo hacia adelante + atrás (estilo Vim)
    drop        = 0.1
    # --- entrenamiento ---
    epochs      = 5
    batch_size  = 64
    lr          = 3e-4
    weight_decay= 0.05
    label_smooth= 0.1
    num_workers = 2
    amp         = True     # mixed precision (fp16)
    seed        = 42

cfg = CFG()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(cfg.seed)
print('device =', device)""")

# ----------------------------------------------------------------------------
md(r"""## 3. Datasets y DataLoaders

`ImageFolder` toma cada subcarpeta como una clase (orden alfabético). Aumentos de datos suaves en *train*; *val/test* sin aumentos.""")

code("""import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

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

train_ds = ImageFolder(os.path.join(cfg.data_root, 'Train'), transform=train_tf)
val_ds   = ImageFolder(os.path.join(cfg.data_root, 'Val'),   transform=eval_tf)
test_ds  = ImageFolder(os.path.join(cfg.data_root, 'Test'),  transform=eval_tf)

classes = train_ds.classes
num_classes = len(classes)
print('Clases:', num_classes)
print('Train/Val/Test:', len(train_ds), len(val_ds), len(test_ds))

train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                      num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
val_dl   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False,
                      num_workers=cfg.num_workers, pin_memory=True)
test_dl  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False,
                      num_workers=cfg.num_workers, pin_memory=True)""")

md(r"""### 3.1 Vistazo a las imágenes y al balance de clases""")

code("""import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# distribución de clases en train
counts = Counter([train_ds.targets[i] for i in range(len(train_ds))])
plt.figure(figsize=(12,4))
plt.bar(range(num_classes), [counts[i] for i in range(num_classes)])
plt.xticks(range(num_classes), classes, rotation=90, fontsize=7)
plt.title('Imágenes por clase (Train)'); plt.tight_layout(); plt.show()

# grilla de ejemplos
def denorm(t):
    t = t.clone()
    for c in range(3):
        t[c] = t[c]*IMAGENET_STD[c] + IMAGENET_MEAN[c]
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
Imagen (3, 160, 160)
        │
        ▼
┌──────────────────────────┐
│ PatchEmbed                │  Conv2d 16x16 stride16 -> (100, 192)
│ + Positional Embedding    │  + tokens posicionales aprendidos
└──────────────────────────┘
        │
        ▼   x N bloques  (residual + pre-norm)
┌──────────────────────────┐
│  RMSNorm                  │
│  MambaBlock (S6)          │   <- el corazón del modelo
│    in_proj -> (x, z)      │   proyección + rama de compuerta
│    Conv1d causal + SiLU   │   mezcla local de tokens
│    SSM selectivo:         │
│      Δ,B,C = f(x)         │   parámetros dependientes de la entrada
│      A = -exp(A_log)      │   matriz de estado (estable)
│      scan: h=Āh+B̄x ; y=Ch │   recurrencia O(L)
│    y = y * SiLU(z)        │   compuerta multiplicativa
│    out_proj               │
└──────────────────────────┘
        │
        ▼
┌──────────────────────────┐
│ RMSNorm + Mean Pooling    │
│ Linear -> 29 clases       │
└──────────────────────────┘
```

**Por qué cada pieza:**
- **PatchEmbed:** convierte la imagen en una *secuencia* de 100 vectores (tokens), como hace ViT. Mamba necesita una secuencia.
- **RMSNorm:** normalización ligera (sin media), estándar en Mamba.
- **in_proj → (x, z):** expande el canal y crea una rama `z` que servirá de **compuerta** (gating, idea de los SiLU/GLU).
- **Conv1d causal:** da contexto **local** entre tokens vecinos antes del SSM (Mamba combina conv local + SSM global).
- **SSM selectivo (S6):** $\Delta, B, C$ se **calculan desde la entrada** → el modelo filtra información de forma dinámica. El *scan* propaga el estado $h$ a lo largo de la secuencia en tiempo lineal.
- **Bidireccional:** como una imagen no tiene un "orden causal" natural, escaneamos hacia adelante **y** hacia atrás (idea de *Vision Mamba / Vim*).
- **Compuerta `y * SiLU(z)`:** controla cuánta señal del SSM pasa.
- **Mean pooling + Linear:** promedia los tokens y clasifica.""")

code(r'''import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))
    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class MambaBlock(nn.Module):
    """Bloque S6 (selective SSM) implementado en PyTorch puro."""
    def __init__(self, cfg):
        super().__init__()
        self.d_model = cfg.embed_dim
        self.d_inner = cfg.expand * cfg.embed_dim
        self.n       = cfg.d_state
        self.dt_rank = cfg.dt_rank
        self.bidirectional = cfg.bidirectional

        # in_proj: produce x y la compuerta z
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)

        # conv1d causal depthwise (contexto local)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner,
                                kernel_size=cfg.conv_kernel,
                                groups=self.d_inner,
                                padding=cfg.conv_kernel - 1, bias=True)

        # proyección que genera (Delta, B, C) dependientes de la entrada
        self.x_proj  = nn.Linear(self.d_inner, self.dt_rank + 2 * self.n, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # parámetros del SSM: A (estable via -exp) y D (skip)
        A = torch.arange(1, self.n + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))         # (d_inner, n)
        self.D     = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def selective_scan(self, u, delta, A, B, C, D):
        # u, delta: (b, l, d_inner) | A: (d_inner, n) | B, C: (b, l, n) | D: (d_inner,)
        b, l, d_in = u.shape
        n = A.shape[1]
        h = u.new_zeros(b, d_in, n)
        ys = []
        for t in range(l):
            dt   = delta[:, t]                                   # (b, d_inner)
            dA   = torch.exp(dt.unsqueeze(-1) * A)               # (b, d_inner, n)
            dBu  = (dt.unsqueeze(-1) * B[:, t].unsqueeze(1)) * u[:, t].unsqueeze(-1)
            h    = dA * h + dBu                                  # estado recurrente
            y    = torch.einsum('bdn,bn->bd', h, C[:, t])        # salida en t
            ys.append(y)
        y = torch.stack(ys, dim=1)                              # (b, l, d_inner)
        return y + u * D

    def ssm(self, x):
        A = -torch.exp(self.A_log.float())                      # (d_inner, n)
        x_dbl = self.x_proj(x)                                  # (b, l, dt_rank+2n)
        delta, B, C = torch.split(x_dbl, [self.dt_rank, self.n, self.n], dim=-1)
        delta = F.softplus(self.dt_proj(delta))                # (b, l, d_inner) > 0
        # el scan se hace en fp32 por estabilidad (exp)
        return self.selective_scan(x.float(), delta.float(), A,
                                   B.float(), C.float(), self.D.float())

    def forward(self, x):
        b, l, d = x.shape
        x_and_z = self.in_proj(x)                              # (b, l, 2*d_inner)
        xs, z = x_and_z.chunk(2, dim=-1)
        # conv1d causal: (b, d_inner, l)
        xs = xs.transpose(1, 2)
        xs = self.conv1d(xs)[..., :l]
        xs = xs.transpose(1, 2)
        xs = F.silu(xs)
        # SSM (forward) + opcional backward
        y = self.ssm(xs)
        if self.bidirectional:
            y = y + self.ssm(xs.flip(1)).flip(1)
        y = y * F.silu(z)                                      # compuerta
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
        x = self.patch(x)                       # (b, D, H', W')
        x = x.flatten(2).transpose(1, 2)        # (b, L, D)
        x = self.drop(x + self.pos)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm_f(x)
        x = x.mean(dim=1)                       # mean pooling de tokens
        return self.head(x)


model = VisionMamba(cfg, num_classes).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(model)
print(f'\nParámetros entrenables: {n_params/1e6:.2f} M')''')

md(r"""### 4.1 Prueba rápida del *forward* (sanity check)""")

code("""model.eval()
with torch.no_grad():
    dummy = torch.randn(2, 3, cfg.img_size, cfg.img_size, device=device)
    out = model(dummy)
print('Entrada :', tuple(dummy.shape))
print('Salida  :', tuple(out.shape), '(batch, num_classes)')
assert out.shape == (2, num_classes)
print('OK ✔')""")

md(r"""### 4.2 Topología del modelo (resumen de capas)
El profesor sugirió mostrar la topología de la red. `torchinfo` lista cada capa, su forma de salida y sus parámetros.""")

code("""!pip install -q torchinfo
from torchinfo import summary
summary(model, input_size=(1, 3, cfg.img_size, cfg.img_size),
        col_names=['input_size', 'output_size', 'num_params'], depth=3)""")

# ----------------------------------------------------------------------------
md(r"""## 5. Entrenamiento

- Optimizador **AdamW** + *cosine schedule*.
- **Label smoothing** 0.1 (regulariza).
- **Mixed precision** (AMP) para ahorrar memoria y acelerar; el *scan* se fuerza a fp32 internamente.""")

code("""from torch.amp import autocast, GradScaler

criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smooth)
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs * len(train_dl))
scaler = GradScaler(enabled=cfg.amp)

@torch.no_grad()
def evaluate(dl):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        with autocast('cuda', enabled=cfg.amp):
            out = model(x)
            loss = criterion(out, y)
        loss_sum += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total""")

code("""import time

history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
best_acc = 0.0

for epoch in range(cfg.epochs):
    model.train()
    t0 = time.time()
    run_loss = run_correct = run_total = 0
    for it, (x, y) in enumerate(train_dl):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast('cuda', enabled=cfg.amp):
            out = model(x)
            loss = criterion(out, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        run_loss    += loss.item() * x.size(0)
        run_correct += (out.argmax(1) == y).sum().item()
        run_total   += x.size(0)
        if it % 100 == 0:
            print(f'  ep{epoch+1} it{it}/{len(train_dl)} '
                  f'loss {run_loss/run_total:.3f} acc {run_correct/run_total:.3f}')

    tr_loss, tr_acc = run_loss/run_total, run_correct/run_total
    va_loss, va_acc = evaluate(val_dl)
    history['train_loss'].append(tr_loss); history['train_acc'].append(tr_acc)
    history['val_loss'].append(va_loss);   history['val_acc'].append(va_acc)
    print(f'[Época {epoch+1}/{cfg.epochs}] '
          f'train_loss {tr_loss:.3f} acc {tr_acc:.3f} | '
          f'val_loss {va_loss:.3f} acc {va_acc:.3f} | '
          f'{time.time()-t0:.0f}s')

    if va_acc > best_acc:
        best_acc = va_acc
        torch.save({'model': model.state_dict(), 'classes': classes,
                    'cfg': vars(cfg)}, '/content/vision_mamba_best.pt')
        print(f'  >> mejor modelo guardado (val_acc {best_acc:.3f})')

print('\\nMejor val_acc:', round(best_acc, 4))""")

md(r"""### 5.1 Curvas de entrenamiento""")

code("""ep = range(1, len(history['train_loss'])+1)
plt.figure(figsize=(11,4))
plt.subplot(1,2,1)
plt.plot(ep, history['train_loss'], '-o', label='train')
plt.plot(ep, history['val_loss'], '-o', label='val')
plt.title('Loss'); plt.xlabel('época'); plt.legend(); plt.grid(alpha=.3)
plt.subplot(1,2,2)
plt.plot(ep, history['train_acc'], '-o', label='train')
plt.plot(ep, history['val_acc'], '-o', label='val')
plt.title('Accuracy'); plt.xlabel('época'); plt.legend(); plt.grid(alpha=.3)
plt.tight_layout(); plt.show()""")

# ----------------------------------------------------------------------------
md(r"""## 6. Evaluación en el conjunto de Test

Cargamos el **mejor** checkpoint y medimos en *test* (datos nunca vistos).""")

code("""ckpt = torch.load('/content/vision_mamba_best.pt', map_location=device)
model.load_state_dict(ckpt['model'])
test_loss, test_acc = evaluate(test_dl)
acc_scratch = test_acc   # guardamos para comparar con la Parte B
print(f'TEST  loss {test_loss:.3f}  |  accuracy {test_acc:.4f}')""")

code("""from sklearn.metrics import classification_report, confusion_matrix

model.eval()
all_pred, all_true = [], []
with torch.no_grad():
    for x, y in test_dl:
        x = x.to(device)
        with autocast('cuda', enabled=cfg.amp):
            out = model(x)
        all_pred.extend(out.argmax(1).cpu().tolist())
        all_true.extend(y.tolist())

print(classification_report(all_true, all_pred, target_names=classes, digits=3))""")

md(r"""### 6.1 Matriz de confusión""")

code("""cm = confusion_matrix(all_true, all_pred)
plt.figure(figsize=(11,9))
plt.imshow(cm, cmap='Blues')
plt.colorbar(fraction=0.046)
plt.xticks(range(num_classes), classes, rotation=90, fontsize=6)
plt.yticks(range(num_classes), classes, fontsize=6)
plt.xlabel('Predicho'); plt.ylabel('Real')
plt.title(f'Matriz de confusión — Test (acc {test_acc:.3f})')
plt.tight_layout(); plt.show()""")

md(r"""### 6.2 Predicciones de ejemplo""")

code("""imgs, labels = next(iter(test_dl))
with torch.no_grad():
    with autocast('cuda', enabled=cfg.amp):
        preds = model(imgs.to(device)).argmax(1).cpu()

plt.figure(figsize=(13,8))
for i in range(12):
    plt.subplot(3,4,i+1); plt.imshow(denorm(imgs[i])); plt.axis('off')
    ok = preds[i] == labels[i]
    plt.title(f'real: {classes[labels[i]]}\\npred: {classes[preds[i]]}',
              color='green' if ok else 'red', fontsize=7)
plt.tight_layout(); plt.show()""")

# ----------------------------------------------------------------------------
md(r"""# PARTE B — Transfer Learning con un Mamba preentrenado

> Esto es lo que el profesor pidió: *"vamos a hacer un re-entrenamiento de la red... se le quitan las entradas y las salidas y se le ponen las de uno"*.

Usamos **MambaVision (NVIDIA, 2024)**, un backbone **Mamba (SSM) para visión** ya entrenado en **ImageNet-1K**. La idea del *transfer learning*:

1. Cargar el modelo **preentrenado** (ya sabe extraer características visuales generales).
2. **Reemplazar la cabeza** (1000 clases de ImageNet → **29 clases** de PlantVillage).
3. **Fine-tuning**: re-entrenar solo unas pocas épocas. Converge rápido y con **mejor accuracy** que entrenar desde cero.

```
ImageNet (1000 clases)              PlantVillage (29 clases)
┌───────────────┐                   ┌───────────────┐
│ Backbone Mamba│  =  se reutiliza  │ Backbone Mamba│  (pesos preentrenados)
│  (preentrenado)│                  │  (preentrenado)│
├───────────────┤                   ├───────────────┤
│ Head -> 1000  │  =  se reemplaza  │ Head -> 29    │  (nueva, se entrena)
└───────────────┘                   └───────────────┘
```""")

md(r"""## 7. Instalar MambaVision

`mamba-ssm` / `causal-conv1d` compilan sus kernels CUDA en Colab (~5-10 min la primera vez). Si la instalación falla por versiones de torch/CUDA, reinicia el entorno y vuelve a ejecutar.""")

code("""!pip install -q mambavision causal-conv1d mamba-ssm transformers timm
print('Instalación lista. Si hubo error de compilación, reinicia el entorno y reejecuta.')""")

md(r"""## 8. Cargar el modelo preentrenado y reemplazar la cabeza""")

code("""import torch.nn as nn
from transformers import AutoModelForImageClassification

# Modelo Mamba preentrenado en ImageNet-1K (variante Tiny, la más liviana)
mv = AutoModelForImageClassification.from_pretrained(
    'nvidia/MambaVision-T-1K', trust_remote_code=True)

# --- reemplazar la cabeza: 1000 clases (ImageNet) -> 29 (PlantVillage) ---
in_features = mv.model.head.in_features
mv.model.head = nn.Linear(in_features, num_classes)
mv = mv.to(device)

print('Cabeza nueva:', mv.model.head)
print('Parámetros totales:', sum(p.numel() for p in mv.parameters())/1e6, 'M')""")

md(r"""### 8.1 Congelar el backbone (entrenar solo la cabeza primero)
Estrategia robusta: primero entrenamos **solo la cabeza** (backbone congelado) y luego, si hay tiempo, descongelamos para *fine-tuning* completo con lr más bajo.""")

code("""# Congelar todo el backbone, dejar entrenable solo la cabeza
for p in mv.parameters():
    p.requires_grad = False
for p in mv.model.head.parameters():
    p.requires_grad = True

trainable = sum(p.numel() for p in mv.parameters() if p.requires_grad)
print(f'Parámetros entrenables (solo cabeza): {trainable/1e3:.1f} K')""")

md(r"""## 9. DataLoaders a 224×224 (tamaño que espera MambaVision)""")

code("""mv_train_tf = T.Compose([
    T.RandomResizedCrop(224, scale=(0.7, 1.0)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(15),
    T.ColorJitter(0.2, 0.2, 0.2),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
mv_eval_tf = T.Compose([
    T.Resize(256), T.CenterCrop(224),
    T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

mv_train_ds = ImageFolder(os.path.join(cfg.data_root, 'Train'), transform=mv_train_tf)
mv_val_ds   = ImageFolder(os.path.join(cfg.data_root, 'Val'),   transform=mv_eval_tf)
mv_test_ds  = ImageFolder(os.path.join(cfg.data_root, 'Test'),  transform=mv_eval_tf)

mv_train_dl = DataLoader(mv_train_ds, batch_size=32, shuffle=True,
                         num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
mv_val_dl   = DataLoader(mv_val_ds, batch_size=32, shuffle=False,
                         num_workers=cfg.num_workers, pin_memory=True)
mv_test_dl  = DataLoader(mv_test_ds, batch_size=32, shuffle=False,
                         num_workers=cfg.num_workers, pin_memory=True)
print('DataLoaders 224 listos.')""")

md(r"""## 10. Fine-tuning

MambaVision devuelve un objeto con `.logits`. Definimos un paso de entrenamiento/evaluación que lo contempla.""")

code("""def mv_logits(out):
    # AutoModelForImageClassification devuelve un objeto con .logits
    return out.logits if hasattr(out, 'logits') else out

@torch.no_grad()
def mv_evaluate(dl):
    mv.eval()
    correct = total = 0
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        with autocast('cuda', enabled=cfg.amp):
            logits = mv_logits(mv(x))
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return correct / total""")

code("""# Fase 1: entrenar solo la cabeza (backbone congelado)
EPOCHS_HEAD = 2
opt = torch.optim.AdamW([p for p in mv.parameters() if p.requires_grad],
                        lr=1e-3, weight_decay=cfg.weight_decay)
scaler_mv = GradScaler(enabled=cfg.amp)
crit = nn.CrossEntropyLoss(label_smoothing=cfg.label_smooth)

for epoch in range(EPOCHS_HEAD):
    mv.train(); t0 = time.time(); run_c = run_t = 0
    for it, (x, y) in enumerate(mv_train_dl):
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        with autocast('cuda', enabled=cfg.amp):
            logits = mv_logits(mv(x))
            loss = crit(logits, y)
        scaler_mv.scale(loss).backward()
        scaler_mv.step(opt); scaler_mv.update()
        run_c += (logits.argmax(1) == y).sum().item(); run_t += x.size(0)
        if it % 100 == 0:
            print(f'  [cabeza] ep{epoch+1} it{it}/{len(mv_train_dl)} acc {run_c/run_t:.3f}')
    print(f'[Cabeza {epoch+1}/{EPOCHS_HEAD}] train_acc {run_c/run_t:.3f} | '
          f'val_acc {mv_evaluate(mv_val_dl):.3f} | {time.time()-t0:.0f}s')""")

md(r"""### 10.1 (Opcional) Descongelar y *fine-tuning* completo
Descongelamos todo el modelo y entrenamos 1-2 épocas más con un learning rate **bajo** (para no destruir lo aprendido en ImageNet).""")

code("""# Fase 2: fine-tuning completo (lr bajo)
for p in mv.parameters():
    p.requires_grad = True

EPOCHS_FT = 1
opt = torch.optim.AdamW(mv.parameters(), lr=2e-5, weight_decay=cfg.weight_decay)
scaler_mv = GradScaler(enabled=cfg.amp)

for epoch in range(EPOCHS_FT):
    mv.train(); t0 = time.time(); run_c = run_t = 0
    for it, (x, y) in enumerate(mv_train_dl):
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        with autocast('cuda', enabled=cfg.amp):
            logits = mv_logits(mv(x))
            loss = crit(logits, y)
        scaler_mv.scale(loss).backward()
        scaler_mv.step(opt); scaler_mv.update()
        run_c += (logits.argmax(1) == y).sum().item(); run_t += x.size(0)
        if it % 100 == 0:
            print(f'  [full] ep{epoch+1} it{it}/{len(mv_train_dl)} acc {run_c/run_t:.3f}')
    print(f'[Full {epoch+1}/{EPOCHS_FT}] train_acc {run_c/run_t:.3f} | '
          f'val_acc {mv_evaluate(mv_val_dl):.3f} | {time.time()-t0:.0f}s')""")

md(r"""## 11. Evaluación de MambaVision en Test y comparación""")

code("""acc_mv = mv_evaluate(mv_test_dl)
print(f'TEST accuracy MambaVision (transfer learning): {acc_mv:.4f}')
print(f'TEST accuracy Vision Mamba (from scratch)    : {acc_scratch:.4f}')

plt.figure(figsize=(5,4))
plt.bar(['Mamba\\nfrom scratch', 'MambaVision\\ntransfer learning'],
        [acc_scratch, acc_mv], color=['#888', '#2a9d8f'])
plt.ylabel('Test accuracy'); plt.ylim(0,1)
for i,v in enumerate([acc_scratch, acc_mv]):
    plt.text(i, v+0.02, f'{v:.3f}', ha='center')
plt.title('Comparación de los dos enfoques'); plt.tight_layout(); plt.show()""")

# ----------------------------------------------------------------------------
md(r"""## 12. Guardar resultados en Drive""")

code("""# from-scratch
shutil.copy('/content/vision_mamba_best.pt', '/content/drive/MyDrive/vision_mamba_best.pt')
# transfer learning
torch.save({'model': mv.state_dict(), 'classes': classes},
           '/content/drive/MyDrive/mambavision_finetuned.pt')
print('Modelos guardados en Drive:')
print(' - MyDrive/vision_mamba_best.pt        (from scratch)')
print(' - MyDrive/mambavision_finetuned.pt    (transfer learning)')""")

md(r"""## 13. Conclusiones y puntos para la presentación

**Lo que mostramos (los dos enfoques):**
- **Parte A:** implementamos **Vision Mamba desde cero** (bloque S6 con *selective scan*) → demuestra que entendemos **cada capa** (Punto 1).
- **Parte B:** **transfer learning** con **MambaVision** preentrenado → es lo que pidió el profesor (re-entrenamiento), converge rápido y da **mejor accuracy** con menos épocas.
- Mamba reemplaza la **atención $O(L^2)$** del Transformer por una **recurrencia SSM $O(L)$**: escala linealmente con la longitud de la secuencia.

**Ideas clave de la arquitectura (Punto 1):**
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

**Mamba vs Transformer vs CNN:**
- **CNN:** muy buen sesgo local, pero campo receptivo limitado.
- **Transformer/ViT:** contexto global, pero costo cuadrático en memoria/cómputo.
- **Mamba:** contexto global con **costo lineal** y *memoria selectiva* → eficiente en secuencias largas.

**Limitaciones de esta demo:**
- El *scan* en PyTorch puro es lento (sin los kernels CUDA de `mamba-ssm`); en producción se usarían esos kernels.
- Modelo pequeño y pocas épocas: la accuracy puede subir con más profundidad/épocas.
""")

# ----------------------------------------------------------------------------
# Convertir source (str) -> lista de líneas con saltos, como pide nbformat
for c in cells:
    s = c["source"]
    c["source"] = s.splitlines(keepends=True)

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

out = "Vision_Mamba_PlantVillage.ipynb"
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("Notebook escrito:", out, "| celdas:", len(cells))
