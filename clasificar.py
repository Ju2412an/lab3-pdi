# -*- coding: utf-8 -*-
"""Clasifica UNA imagen que tú elijas con el Vision Mamba ya entrenado.

Dos formas de usarlo:
  1) Con selector de archivos (haz click y elige la imagen):
        .venv\\Scripts\\python.exe clasificar.py
  2) Pasando la ruta directamente:
        .venv\\Scripts\\python.exe clasificar.py "ruta\\a\\mi_hoja.jpg"

Muestra la imagen, la clase predicha y el top-5 de probabilidades.
"""
import os, sys, torch
import torch.nn.functional as F
import train_local as m
import torchvision.transforms as T
from torch.amp import autocast
from PIL import Image
import matplotlib.pyplot as plt

cfg = m.cfg
device = 'cuda' if torch.cuda.is_available() else 'cpu'
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def elegir_imagen():
    """Devuelve la ruta de la imagen: del argumento, o con un selector de archivos."""
    if len(sys.argv) > 1:
        return sys.argv[1]
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        ruta = filedialog.askopenfilename(
            title='Elige una imagen de hoja para clasificar',
            filetypes=[('Imágenes', '*.jpg *.jpeg *.png *.JPG *.JPEG *.PNG'), ('Todos', '*.*')])
        root.destroy()
        return ruta
    except Exception as e:
        print('No se pudo abrir el selector de archivos:', e)
        print('Pasa la ruta como argumento:  python clasificar.py "ruta\\a\\imagen.jpg"')
        return ''


def main():
    ruta = elegir_imagen()
    if not ruta:
        print('No se eligió ninguna imagen.'); return
    if not os.path.exists(ruta):
        print('No existe el archivo:', ruta); return

    # --- cargar modelo entrenado ---
    ckpt = torch.load('results/vision_mamba_best.pt', map_location=device)
    classes = ckpt['classes']
    model = m.VisionMamba(cfg, len(classes)).to(device)
    model.load_state_dict(ckpt['model']); model.eval()

    # --- preprocesar la imagen elegida ---
    eval_tf = T.Compose([T.Resize(int(cfg.img_size * 1.14)), T.CenterCrop(cfg.img_size),
                         T.ToTensor(), T.Normalize(MEAN, STD)])
    img = Image.open(ruta).convert('RGB')
    x = eval_tf(img).unsqueeze(0).to(device)

    # --- predecir ---
    with torch.no_grad():
        with autocast('cuda', enabled=(device == 'cuda')):
            logits = model(x)
    probs = F.softmax(logits.float(), dim=1)[0].cpu()
    top_p, top_i = probs.topk(5)

    print(f'\nImagen: {ruta}')
    print(f'Predicción: {classes[top_i[0]]}  ({top_p[0]*100:.1f}%)\n')
    print('Top-5:')
    for p, i in zip(top_p, top_i):
        print(f'  {classes[i]:38s} {p*100:5.1f}%')

    # --- mostrar imagen + barras del top-5 ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.imshow(img); ax1.axis('off')
    ax1.set_title(f'{classes[top_i[0]]}\n({top_p[0]*100:.1f}%)',
                  color='green', fontsize=13)
    nombres = [classes[i] for i in top_i][::-1]
    valores = [float(p) * 100 for p in top_p][::-1]
    ax2.barh(nombres, valores, color='#2a9d8f')
    ax2.set_xlabel('probabilidad (%)'); ax2.set_xlim(0, 100)
    ax2.set_title('Top-5 predicciones')
    for j, v in enumerate(valores):
        ax2.text(v + 1, j, f'{v:.1f}%', va='center', fontsize=9)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
