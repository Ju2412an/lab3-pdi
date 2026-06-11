# -*- coding: utf-8 -*-
"""Interfaz web (Gradio) para la demo de Vision Mamba — PlantVillage.

Pestañas:
  1. Resultados del entrenamiento  — curvas, matriz de confusión, reporte.
  2. Clasificar imagen             — sube/elige una hoja y obtén el top-5.
  3. Prueba aleatoria del Test     — N imágenes del Test con sus predicciones.

Uso:
    .venv\\Scripts\\python.exe app.py
Luego abre en el navegador la dirección que aparece (http://127.0.0.1:7860).
"""
import os, torch
import torch.nn.functional as F
import gradio as gr
import train_local as m
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.amp import autocast
from PIL import Image

cfg = m.cfg
device = 'cuda' if torch.cuda.is_available() else 'cpu'
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
eval_tf = T.Compose([T.Resize(int(cfg.img_size * 1.14)), T.CenterCrop(cfg.img_size),
                     T.ToTensor(), T.Normalize(MEAN, STD)])

# --- cargar modelo entrenado una sola vez ---
ckpt = torch.load('results/vision_mamba_best.pt', map_location=device)
classes = ckpt['classes']
model = m.VisionMamba(cfg, len(classes)).to(device)
model.load_state_dict(ckpt['model'])
model.eval()

# --- dataset de Test (para la pestaña de prueba aleatoria y ejemplos) ---
test_ds = ImageFolder(os.path.join(cfg.data_root, 'Test'), transform=eval_tf)
GEN = torch.Generator().manual_seed(7)

# accuracy de referencia (de la corrida)
import json
try:
    ACC = json.load(open('results/resumen.json', encoding='utf-8'))['test_acc']
except Exception:
    ACC = None


def predecir_pil(img: Image.Image):
    """Devuelve un dict {clase: probabilidad} con el top-5 (formato gr.Label)."""
    x = eval_tf(img.convert('RGB')).unsqueeze(0).to(device)
    with torch.no_grad():
        with autocast('cuda', enabled=(device == 'cuda')):
            probs = F.softmax(model(x).float(), dim=1)[0].cpu()
    top_p, top_i = probs.topk(5)
    return {classes[int(i)]: float(p) for p, i in zip(top_p, top_i)}


def prueba_aleatoria(n):
    """Toma n imágenes aleatorias del Test y arma una galería con su resultado."""
    n = int(n)
    idx = torch.randperm(len(test_ds), generator=GEN)[:n].tolist()
    paths = [test_ds.samples[i][0] for i in idx]
    labels = [test_ds.samples[i][1] for i in idx]
    imgs = torch.stack([test_ds[i][0] for i in idx]).to(device)
    with torch.no_grad():
        with autocast('cuda', enabled=(device == 'cuda')):
            preds = model(imgs).argmax(1).cpu().tolist()
    galeria, aciertos = [], 0
    for p, lab, path in zip(preds, labels, paths):
        ok = p == lab
        aciertos += ok
        marca = '✅' if ok else '❌'
        cap = f'{marca} real: {classes[lab]} | pred: {classes[p]}'
        galeria.append((path, cap))
    resumen = f'### Aciertos: {aciertos}/{n}  ({aciertos/n*100:.0f}%)'
    return galeria, resumen


# ---------------------------------------------------------------------------
# Imágenes de ejemplo (alta confianza, elegidas a mano) para un clic en la presentación
_EJEMPLOS_HC = [
    ('Peach - Bacterial Spot',     '7860212d-e49b-4ea7-bbc4-4df7d0484640___Rutg._Bact.S 1613.JPG'),
    ('Corn (Maize) - Common Rust', 'RS_Rust 2667.JPG'),
    ('Cherry - Healthy',           '84807b4a-f76a-4ebf-a78a-ec9444197c55___JR_HL 9786_flipTB.JPG'),
    ('Grape - Leaf Blight',        '2dfd88f8-1ddf-49db-b4be-2929275a0629___FAM_L.Blight 1430.JPG'),
    ('Strawberry - Leaf Scorch',   '84ee727c-d814-4f2c-8afc-5564ecb4e1a0___RS_L.Scorch 0953_flipLR.JPG'),
]
EJEMPLOS = []
for sub, fname in _EJEMPLOS_HC:
    p = os.path.join(cfg.data_root, 'Test', sub, fname)
    if os.path.exists(p):
        EJEMPLOS.append(p)
    else:  # respaldo: primera imagen de esa clase
        d = os.path.join(cfg.data_root, 'Test', sub)
        fs = sorted(os.listdir(d)) if os.path.isdir(d) else []
        if fs:
            EJEMPLOS.append(os.path.join(d, fs[0]))

acc_txt = f'**Accuracy en Test: {ACC*100:.1f}%**' if ACC else ''

with gr.Blocks(title='Vision Mamba · PlantVillage', theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"""# 🌿 Vision Mamba (SSM) — Clasificación de enfermedades en plantas
Tarea 3 · Procesamiento de Imágenes · Equipo 2 — modelo **Mamba implementado desde cero**
(29 clases · {acc_txt})""")

    with gr.Tab('📊 Resultados del entrenamiento'):
        gr.Markdown('Curvas de entrenamiento, matriz de confusión y ejemplos de la corrida.')
        with gr.Row():
            gr.Image('results/curvas.png', label='Curvas (loss / accuracy)')
            gr.Image('results/matriz_confusion.png', label='Matriz de confusión (Test)')
        with gr.Row():
            gr.Image('results/predicciones.png', label='Predicciones de ejemplo')
            gr.Image('results/clases_balance.png', label='Imágenes por clase')
        try:
            with open('results/classification_report.txt', encoding='utf-8') as f:
                rep = f.read()
            gr.Code(rep, label='Reporte de clasificación (precision / recall / F1)')
        except Exception:
            pass

    with gr.Tab('🔎 Clasificar imagen'):
        gr.Markdown('Sube o arrastra una hoja (o usa un ejemplo). El modelo da el **top-5**.')
        with gr.Row():
            inp = gr.Image(type='pil', label='Imagen de la hoja', height=320)
            out = gr.Label(num_top_classes=5, label='Predicción (top-5)')
        btn = gr.Button('Clasificar', variant='primary')
        btn.click(predecir_pil, inputs=inp, outputs=out)
        inp.change(predecir_pil, inputs=inp, outputs=out)
        if EJEMPLOS:
            gr.Examples(examples=[[e] for e in EJEMPLOS], inputs=inp,
                        label='Ejemplos de alta confianza (clic para probar)')

    with gr.Tab('🎲 Prueba aleatoria del Test'):
        gr.Markdown('Toma imágenes al azar del conjunto de prueba y muestra sus predicciones.')
        n = gr.Slider(4, 16, value=8, step=1, label='Número de imágenes')
        btn2 = gr.Button('Probar', variant='primary')
        res = gr.Markdown()
        gal = gr.Gallery(label='Resultados', columns=4, height=420)
        btn2.click(prueba_aleatoria, inputs=n, outputs=[gal, res])

if __name__ == '__main__':
    demo.launch(inbrowser=True)
