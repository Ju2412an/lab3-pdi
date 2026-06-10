# Vision Mamba (SSM) — Clasificación de enfermedades en plantas 🌿

**Tarea 3 · Procesamiento de Imágenes (UDEA) · Equipo 2**
Implementación de **Mamba (modelo de espacio de estados, 2023)** para visión, clasificando el
dataset **PlantVillage (29 clases)**. Versión **LOCAL** (Windows + GPU NVIDIA), adaptada desde
el notebook original de Google Colab.

> **Resultado de referencia (ya generado):** **78.5 % de accuracy en Test** con un Vision Mamba
> *desde cero* (0.85 M parámetros, 5 épocas). Las figuras están en [`results/`](results/) y se
> ven directamente en GitHub, sin necesidad de ejecutar nada.

---

## 🚀 TL;DR para ejecutarlo (Windows + GPU NVIDIA)

```powershell
# 1. Clona el repo y entra en la carpeta
git clone <URL-DEL-REPO>.git
cd "practica 3"

# 2. Coloca el dataset:  pon tu  archive.zip  en esta misma carpeta
#    (el script lo descomprime solo a ./data la primera vez)

# 3. Monta el entorno (Python 3.12 + PyTorch CUDA, aislado). Una sola vez:
.\setup_local.ps1

# 4. Entrena y genera resultados en ./results/  (~25 min en una GTX 1060)
.venv\Scripts\python.exe train_local.py
```

Si PowerShell bloquea el script, ábrelo así una vez:
`powershell -ExecutionPolicy Bypass -File .\setup_local.ps1`

---

## 📁 Qué hay en el repo

| Archivo | Para qué |
|---|---|
| `train_local.py` | **Script principal** (Parte A: Vision Mamba desde cero). Entrena y guarda todo en `results/`. |
| `Vision_Mamba_PlantVillage_LOCAL.ipynb` | El mismo contenido en **notebook**, para exponer celda por celda. |
| `build_notebook_local.py` | Regenera el notebook anterior. |
| `setup_local.ps1` | Monta el entorno automáticamente (uv + Python 3.12 + PyTorch CUDA). |
| `requirements.txt` | Dependencias (alternativa manual al script de setup). |
| `results/` | Figuras y métricas **ya generadas** (curvas, matriz de confusión, predicciones, reporte). |
| `results/vision_mamba_best.pt` | Checkpoint del mejor modelo (para evaluar sin re-entrenar). |
| `GUION_PRESENTACION.md`, `GUION_APERTURA.md` | Guiones de la exposición. |
| `Vision_Mamba_PlantVillage.ipynb` | Notebook original de Colab (referencia). |

**No están en el repo** (por tamaño, ver `.gitignore`): el dataset (`archive.zip`, `data/`), el
entorno (`.venv/`) y las grabaciones de clase. El dataset lo aporta cada quien.

---

## 🖥️ Requisitos

- **Windows** con **GPU NVIDIA** (probado en GTX 1060 6 GB, driver CUDA 12.x).
- Python ≥ 3.9 en el sistema (solo para arrancar `uv`; el entorno real usa 3.12 aislado).
- El **dataset PlantVillage** (`archive.zip`) con esta estructura al descomprimir:
  ```
  data/Plant Village Dataset/
  ├── Train/  (53.691 img · 29 clases)
  ├── Val/    (12.067 img)
  └── Test/   ( 1.355 img)
  ```

> **¿Sin GPU NVIDIA?** El modelo corre en CPU pero el *selective scan* en PyTorch puro es
> extremadamente lento. En ese caso **no entrenes**: muestra directamente las figuras de
> `results/` (ya están listas) para la presentación.

---

## ⚙️ Configuración (en `train_local.py`, clase `CFG`)

Pensada para 6 GB de VRAM y una demo rápida:

- `train_per_class = 250` → imágenes por clase para entrenar. Pon `None` para usar **todo** el
  dataset (mucho más lento, mejor accuracy).
- `epochs = 5`, `img_size = 128`, `batch_size = 32` → seguros en 6 GB.
- Cuello de botella: el *selective scan* secuencial (~1.1 s/iter en la 1060), **no** la VRAM
  (pico ~1.4 GB).

---

## 📊 Resultados (corrida de referencia)

- **Test accuracy: 78.5 %** (29 clases) · mejor `val_acc` 77.0 %
- Convergencia limpia, sin overfitting (ver `results/curvas.png`)
- Diagonal fuerte en la matriz de confusión (`results/matriz_confusion.png`)

Para un modelo **sin pesos preentrenados** y entrenado pocos minutos, es un resultado sólido.

---

## 🧩 Parte B — Transfer Learning (opcional)

El profesor pidió un *re-entrenamiento* (Mamba preentrenado + cabeza nueva de 29 clases). Se usaría
**MambaVision (NVIDIA)**, pero requiere compilar los kernels CUDA de `mamba-ssm`/`causal-conv1d`,
algo **poco fiable en Windows + GPU Pascal (GTX 1060)**. Por eso esta parte está al final del
notebook, protegida con `try/except`, y es **opcional**. Para correrla de verdad: **Linux/WSL2**
con una GPU más reciente, o mostrar los resultados de Colab para esa sección.

---

## 📝 Notas

- PyTorch aún no publica ruedas CUDA para Python 3.14; por eso el entorno usa **Python 3.12**.
- El entorno se crea con [`uv`](https://github.com/astral-sh/uv) y queda aislado en `.venv/`
  (no toca tu Python global).
