# Guion — Demostración funcional + Apéndice de código
### Tarea 3 · Mamba (SSM) para visión · PlantVillage · Equipo 2

> **Para quién:** la persona que muestra la parte funcional en la exposición.
> Dos secciones: **(1)** el guion de la demo en vivo (qué decir y qué ejecutar) y
> **(2)** un apéndice para entender el código a grandes rasgos antes de presentar.

---

# PARTE 1 — Guion de la demostración funcional (con la interfaz web)

> Toda la demo se hace desde la **interfaz web (Gradio)**: nada de escribir comandos en vivo.
> Solo se lanza una vez al principio y se navega por pestañas.

## ✅ Antes de empezar (preparación, 2 min antes de exponer)
- [ ] Abrir **PowerShell** en la carpeta del proyecto y **lanzar la interfaz**:
  ```powershell
  cd "D:\Education\UDEA\Electivas\procesamiento imagenes\practica 3"
  .venv\Scripts\python.exe app.py
  ```
  Se abre solo en el navegador (`http://127.0.0.1:7860`). **Déjalo abierto antes de exponer.**
- [ ] Tener lista **1 imagen "rara"** (un screenshot o foto de hoja con fondo distinto) en una
  carpeta fácil de encontrar, para el momento del "límite del modelo".
- [ ] Maximizar la ventana del navegador y verificar que se ven las 3 pestañas.

> **Plan B (si la interfaz no abre):** las figuras ya están en `results\` y se ven en GitHub
> (pestaña del repo). También quedan los scripts `demo.py` y `clasificar.py` como respaldo.

---

## 🎬 Momento 1 — "Entrenamos el modelo localmente"
**🖥️ Pestaña:** **📊 Resultados del entrenamiento**

**🎤 Qué decir:**
> "Implementamos Vision Mamba **desde cero** y lo entrenamos en una GPU local. Aquí están las
> curvas: la *accuracy* sube y la *loss* baja de forma estable, y la curva de validación sigue a
> la de entrenamiento, así que **no hay sobreajuste**. En la matriz de confusión se ve una
> **diagonal marcada**: acierta la mayoría de las 29 clases. El resultado final es **78.5% de
> accuracy en Test**, y eso **sin usar pesos preentrenados**." *(Señalar la diagonal y, si
> hay tiempo, bajar al reporte por clase.)*

---

## 🎬 Momento 2 — "Clasifiquemos una hoja enferma, con alta confianza"
**🖥️ Pestaña:** **🔎 Clasificar imagen** → clic en el ejemplo **Peach - Bacterial Spot**

**🎤 Qué decir:**
> "Pasemos a clasificar. Esta es una hoja de durazno con una infección bacteriana; se ven las
> manchas negras. El modelo no solo dice la clase, sino **qué tan seguro está**: la acierta con
> casi **99%**, y el top-5 muestra que las demás opciones quedan muy por debajo."

---

## 🎬 Momento 3 — "Y también reconoce las sanas"
**🖥️ Misma pestaña** → clic en el ejemplo **Cherry - Healthy** (o **Corn - Common Rust**)

**🎤 Qué decir:**
> "No solo detecta enfermedades: con una hoja **sana** también acierta con más del **95%**. Es
> decir, el modelo es **muy confiable** en imágenes parecidas a las de entrenamiento."

---

## 🎬 Momento 4 — "Pero tiene límites" (domain gap)
**🖥️ Misma pestaña** → **arrastrar la imagen "rara"** que preparaste (screenshot / otra fuente)

**🎤 Qué decir:**
> "Ahora le damos una imagen **distinta** a las del dataset. Su confianza cae a ~**28%** y reparte
> la probabilidad entre clases sin relación. Esto se llama **domain gap**: el modelo se entrenó
> con fotos muy estandarizadas (una hoja, fondo uniforme) y **no generaliza** bien a imágenes que
> se ven diferentes. Es una **limitación honesta** que se resolvería con datos más variados."

---

## 🎬 Momento 5 (opcional) — "En conjunto, no fue suerte"
**🖥️ Pestaña:** **🎲 Prueba aleatoria del Test** → dejar 8 imágenes → botón **Probar**

**🎤 Qué decir:**
> "Por último, una prueba al azar sobre imágenes que el modelo **nunca vio**: acierta la mayoría
> (✅), y cuando falla suele confundir **enfermedades parecidas de la misma planta**. Esto
> confirma el 78.5% global; no fue un caso afortunado."

---

## 🏁 Cierre de la parte funcional
**🎤 Qué decir:**
> "En resumen: un Mamba implementado desde cero, entrenado localmente, llega a **78.5%** en 29
> clases de enfermedades de plantas. Es **muy confiable en su dominio** y muestra **límites claros**
> fuera de él. Todo —código, resultados e interfaz— está en nuestro repositorio de GitHub."

**Narrativa en una frase:** *entrenó bien (resultados) → acierta enfermas y sanas con alta
confianza (clasificar) → pero tiene límites fuera de su dominio (imagen rara) → y no fue suerte
(prueba aleatoria).*

---

## 🔁 Resumen de clics (chuleta para tener al lado)
| Paso | Pestaña | Acción |
|---|---|---|
| 1 | 📊 Resultados | señalar curvas + matriz (78.5%) |
| 2 | 🔎 Clasificar | clic ejemplo **Peach - Bacterial Spot** (~99%) |
| 3 | 🔎 Clasificar | clic ejemplo **Cherry - Healthy** (~95%) |
| 4 | 🔎 Clasificar | arrastrar **imagen rara** (~28%, domain gap) |
| 5 | 🎲 Prueba aleatoria | botón **Probar** con 8 imágenes |

---
---

# PARTE 2 — Apéndice: el código a grandes rasgos

> Objetivo: entender **qué hace cada archivo y cada pieza** para responder preguntas, sin
> memorizar línea por línea.

---

## 🖥️ GUÍA DE PANTALLA — qué mostrar mientras tu compañero explica

> **Tu rol:** abrir `train_local.py` y, conforme tu compañero va narrando, **ir saltando a estas
> líneas y señalarlas**. En VS Code puedes saltar con `Ctrl+G` → escribir el número de línea.
> *(El notebook `Vision_Mamba_PlantVillage_LOCAL.ipynb` tiene el mismo código en celdas si lo
> prefieres proyectar así.)*

| # | Cuando tu compañero dice… | Tú muestras (`train_local.py`) | Qué señalar |
|---|---|---|---|
| 1 | "Definimos los hiperparámetros del modelo" | **líneas 24–50** (`class CFG`) | `img_size`, `embed_dim`, `depth`, `d_state`, `epochs` |
| 2 | "La imagen se parte en parches → secuencia de tokens" | **líneas 142–146** (`self.patch`, `self.pos`) | el `Conv2d` 16×16 y el *positional embedding* |
| 3 | "Normalización ligera, estilo Mamba" | **líneas 60–67** (`class RMSNorm`) | que normaliza sin restar la media |
| 4 | "El bloque Mamba expande y crea una compuerta `z`" | **línea 80** + **117–118** (`in_proj`, `chunk`) | cómo sale `(xs, z)` |
| 5 | "Conv1d causal: mezcla cada token con sus vecinos" | **líneas 81–84** + **119–122** | el `Conv1d` *depthwise* + `SiLU` |
| 6 | "Δ, B, C se calculan **desde la entrada** (selectivo)" | **líneas 107–111** (`def ssm`) | `x_proj`, el `split` en Δ,B,C, `softplus` |
| 7 | "A se define estable como −exp(A_log)" | **línea 108** (y def en 87–88) | `A = -torch.exp(self.A_log...)` |
| 8 | "El *selective scan*: recurrencia lineal O(L)" | **líneas 92–105**, núcleo **97–103** | el `for t in range(l)` → `h = dA*h + dBu` |
| 9 | "Escaneamos hacia adelante y atrás (bidireccional)" | **líneas 124–125** | `self.ssm(xs.flip(1)).flip(1)` |
| 10 | "Compuerta: cuánta señal del SSM pasa" | **línea 126** | `y = y * F.silu(z)` |
| 11 | "Se arma el modelo completo: parches → bloques → clase" | **líneas 152–160** (`VisionMamba.forward`) | `patch → blocks → mean(dim=1) → head` |
| 12 | "Promediamos los tokens y clasificamos en 29 clases" | **líneas 159–160** | `x.mean(dim=1)` y `self.head` |
| 13 | "Entrenamos: AdamW, AMP, label smoothing" | **líneas 291–294** | `criterion`, `optimizer`, `scheduler`, `scaler` |
| 14 | "El bucle de entrenamiento por épocas" | **líneas 313–342** | `for epoch …` y `scaler.scale(loss).backward()` (321) |
| 15 | "Evaluamos en Test con el mejor modelo" | **líneas 357–360** | `evaluate(test_dl)` → accuracy |
| 16 | "Matriz de confusión y reporte por clase" | **líneas 371–385** | `classification_report`, `confusion_matrix` |

> **Truco:** ten el archivo abierto ya en la **línea 70** (`class MambaBlock`) antes de empezar la
> Parte 2; ahí está "el corazón" (filas 4–10 de la tabla) y casi todo el peso de la explicación.

---

## A. Archivos del proyecto
| Archivo | Qué hace |
|---|---|
| `train_local.py` | Entrena el modelo y guarda figuras/métricas en `results/`. Es el "todo en uno". |
| `Vision_Mamba_PlantVillage_LOCAL.ipynb` | El mismo contenido en notebook, para exponer por celdas. |
| `demo.py` | Carga el modelo entrenado y muestra 12 predicciones del Test. |
| `clasificar.py` | Clasifica **una** imagen que elijas, con top-5 de probabilidades. |
| `results/` | Curvas, matriz de confusión, predicciones, reporte y el checkpoint `.pt`. |

## B. El flujo general (4 pasos)
1. **Datos:** `ImageFolder` lee las carpetas (cada subcarpeta = una clase). Se aplican
   *transforms* (recorte, volteo, normalización). Para la demo usamos un **subconjunto**
   (250 img/clase) por velocidad.
2. **Modelo:** `VisionMamba` (lo construimos nosotros, ~0.85 M parámetros).
3. **Entrenamiento:** 5 épocas con AdamW, *cosine schedule*, *label smoothing* y *mixed precision*.
   Guardamos el mejor modelo según la *accuracy* de validación.
4. **Evaluación:** cargamos el mejor checkpoint y medimos en Test (datos nunca vistos):
   accuracy, reporte por clase, matriz de confusión, ejemplos.

## C. La arquitectura, pieza por pieza
Una imagen entra y sale una de las 29 clases. El camino:

```
Imagen (3,128,128)
  → PatchEmbed         (Conv2d 16x16)  ⇒ 64 "tokens" de dimensión 160  (+ posición)
  → N× bloques Mamba   (con conexión residual)
  → Mean Pooling + Linear(160 → 29)    ⇒ clase
```

- **PatchEmbed** `(L142–146, 153–154)`**:** parte la imagen en parches de 16×16 y los convierte en
  una **secuencia** de vectores (tokens). Mamba, como los Transformers, trabaja sobre secuencias.
- **MambaBlock (el "S6", el corazón)** `(L70–127)`**:**
  - `in_proj` `(L80, 117–118)`: expande el vector y crea dos ramas: `x` (la señal) y `z` (**compuerta**).
  - `Conv1d causal` `(L81–84, 119–122)`: mezcla cada token con sus **vecinos** (contexto local).
  - **SSM selectivo** `(L107–111)`**:** recurrencia de "espacio de estados" donde los parámetros
    **Δ, B, C se calculan a partir de la entrada** → el modelo **decide qué recordar y qué
    olvidar** en cada paso. Esta es la idea clave de Mamba.
  - **selective scan** `(L92–105)`**:** recorre la secuencia propagando un estado interno `h`. Es una
    recurrencia de coste **lineal** O(L), no cuadrático como la atención de los Transformers.
  - `y * SiLU(z)` `(L126)`: la compuerta controla cuánta señal del SSM pasa.
- **Bidireccional** `(L124–125)`**:** una imagen no tiene "orden" natural, así que escaneamos la
  secuencia **hacia adelante y hacia atrás** y sumamos (idea de *Vision Mamba / Vim*).
- **Mean Pooling + Linear** `(L159–160)`**:** se promedian los tokens y una capa lineal da las 29 clases.

## D. Las 3 ideas que hay que poder defender
1. **Mamba vs Transformer:** Mamba reemplaza la atención **O(L²)** por una recurrencia SSM
   **O(L)** → escala **linealmente** con la longitud de la secuencia.
2. **"Selectivo":** los parámetros del SSM dependen de la entrada (Δ, B, C = f(x)) → memoria
   **dinámica**, el modelo filtra información según lo que ve.
3. **Estabilidad:** la matriz de estado se define como **A = −exp(A_log)**, lo que la mantiene
   estable durante la recurrencia.

## E. Decisiones prácticas (por si preguntan)
- **¿Por qué local y no Colab?** Control total, sin límites de cuota ni de sesión; la GPU local
  alcanza para la Parte A.
- **¿Por qué un subconjunto y pocas épocas?** El *selective scan* está implementado en **PyTorch
  puro** (didáctico, para explicar cada capa), y eso es ~10–20× más lento que los kernels CUDA
  oficiales de Mamba. Con el dataset completo subiría la accuracy pero tardaría mucho más.
- **¿Por qué no hicieron la Parte B (transfer learning con MambaVision)?** Requiere compilar
  kernels CUDA (`mamba-ssm`) que **no son fiables en Windows + GPU Pascal**. Queda documentada y
  es opcional; correría en Linux/WSL2 con una GPU más reciente.

## F. Posibles preguntas del profesor y respuesta corta
- *"¿Esto es Mamba de verdad o un Transformer disfrazado?"* → Es Mamba: no hay atención; el
  mezclado de tokens lo hace el **SSM selectivo + conv1d**, con recurrencia lineal.
- *"¿Por qué la accuracy no es 99%?"* → Modelo pequeño, pocas épocas y subconjunto, todo desde
  cero. Es un compromiso por la demo; el diseño es correcto y escalaría con más datos/épocas.
- *"¿Por qué se confunde con ciertas clases?"* → Entre enfermedades **visualmente parecidas de la
  misma planta** (se ve en la matriz de confusión); es el error esperable.
