# -*- coding: utf-8 -*-
"""Demo rápida: carga el modelo YA entrenado (results/vision_mamba_best.pt)
y muestra predicciones + accuracy en segundos. Ideal para la presentación en vivo,
sin re-entrenar.

Uso:
    .venv\\Scripts\\python.exe demo.py
"""
import os, torch
import train_local as m
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.amp import autocast
import matplotlib.pyplot as plt

cfg = m.cfg
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device =', device)

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
eval_tf = T.Compose([T.Resize(int(cfg.img_size * 1.14)), T.CenterCrop(cfg.img_size),
                     T.ToTensor(), T.Normalize(MEAN, STD)])
test = ImageFolder(os.path.join(cfg.data_root, 'Test'), transform=eval_tf)
classes = test.classes

ckpt = torch.load('results/vision_mamba_best.pt', map_location=device)
model = m.VisionMamba(cfg, len(classes)).to(device)
model.load_state_dict(ckpt['model']); model.eval()
print('Modelo cargado:', sum(p.numel() for p in model.parameters()) / 1e6, 'M parámetros')

# 12 imágenes aleatorias y variadas
g = torch.Generator().manual_seed(7)
idx = torch.randperm(len(test), generator=g)[:12].tolist()
imgs = torch.stack([test[i][0] for i in idx])
labs = [test[i][1] for i in idx]
with torch.no_grad():
    with autocast('cuda', enabled=(device == 'cuda')):
        preds = model(imgs.to(device)).argmax(1).cpu().tolist()

aciertos = sum(p == l for p, l in zip(preds, labs))
print(f'Aciertos en la muestra: {aciertos}/12')


def denorm(t):
    t = t.clone()
    for c in range(3):
        t[c] = t[c] * STD[c] + MEAN[c]
    return t.clamp(0, 1).permute(1, 2, 0).numpy()


plt.figure(figsize=(13, 8))
for i in range(12):
    plt.subplot(3, 4, i + 1); plt.imshow(denorm(imgs[i])); plt.axis('off')
    ok = preds[i] == labs[i]
    plt.title(f'real: {classes[labs[i]]}\npred: {classes[preds[i]]}',
              color='green' if ok else 'red', fontsize=8)
plt.suptitle(f'Vision Mamba — predicciones en Test ({aciertos}/12 correctas)', fontsize=12)
plt.tight_layout()
plt.show()   # abre una ventana con las predicciones
