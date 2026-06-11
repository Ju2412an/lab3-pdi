# Guion — Demostración funcional + Apéndice de código
### Tarea 3 · Mamba (SSM) para visión · PlantVillage · Equipo 2

> **Para quién:** la persona que muestra la parte funcional en la exposición.
> Dos secciones: **(1)** el guion de la demo en vivo (qué decir y qué ejecutar) y
> **(2)** un apéndice para entender el código a grandes rasgos antes de presentar.

---

# PARTE 1 — Guion de la demostración funcional (en vivo)

## ✅ Antes de empezar (preparación, 2 min antes de exponer)
- [ ] Abrir **PowerShell** en la carpeta del proyecto:
  ```powershell
  cd "D:\Education\UDEA\Electivas\procesamiento imagenes\practica 3"
  ```
- [ ] Verificar que la GPU responde (opcional):
  ```powershell
  .venv\Scripts\python.exe -c "import torch; print(torch.cuda.get_device_name(0))"
  ```
- [ ] Tener a mano **2 imágenes**: una hoja del Test (clara) y una imagen "rara"
  (screenshot o foto con fondo distinto) para mostrar el límite del modelo.
- [ ] Tener abierta la carpeta `results\` por si hay que mostrar las figuras de respaldo.

> **Plan B (si algo falla en vivo):** abrir directamente las imágenes de `results\`
> (`curvas.png`, `matriz_confusion.png`, `predicciones.png`). Ya están generadas, no
> dependen de ejecutar nada.

---

## 🎬 Momento 1 — "Entrenamos el modelo localmente"
**🖥️ Mostrar:** `results\curvas.png` y `results\matriz_confusion.png`

**🎤 Qué decir:**
> "Implementamos Vision Mamba **desde cero** y lo entrenamos en una GPU local. Aquí están
> las curvas: la *accuracy* sube y la *loss* baja de forma estable, y la curva de validación
> sigue a la de entrenamiento, así que **no hay sobreajuste**. En la matriz de confusión se ve
> una **diagonal marcada**: el modelo acierta la mayoría de las 29 clases. El resultado final
> es **78.5% de accuracy en Test**, y eso **sin usar pesos preentrenados**."

---

## 🎬 Momento 2 — "Veámoslo prediciendo" (demo.py)
**🖥️ Ejecutar:**
```powershell
.venv\Scripts\python.exe demo.py
```
Se abre una ventana con 12 predicciones del Test (verde = acierto, rojo = error).

**🎤 Qué decir:**
> "Esto son 12 imágenes de prueba que el modelo **nunca vio**. En verde los aciertos, en rojo
> los errores. Acierta **9 de 12**, consistente con el 78.5% global. Y fíjense que cuando se
> equivoca, suele confundir **enfermedades parecidas de la misma planta**, no cosas absurdas."

---

## 🎬 Momento 3 — "Clasifiquemos una hoja en vivo, con alta confianza" (clasificar.py)
**🖥️ Ejecutar** y elegir en el selector una hoja **del Test**:
```powershell
.venv\Scripts\python.exe clasificar.py
```

**🎤 Qué decir:**
> "Ahora una sola imagen. El modelo no solo dice la clase, sino **qué tan seguro está**: aquí la
> predice con más del **90%** de confianza, y el top-5 muestra que las demás opciones quedan muy
> por debajo. En imágenes parecidas a las de entrenamiento, el modelo es **muy confiable**."

---

## 🎬 Momento 4 — "Pero tiene límites" (clasificar.py con imagen rara)
**🖥️ Ejecutar** y elegir una imagen **distinta** (screenshot / otra fuente):
```powershell
.venv\Scripts\python.exe clasificar.py
```

**🎤 Qué decir:**
> "Si en cambio le damos una imagen **distinta** a las del dataset, su confianza cae a ~**28%** y
> reparte la probabilidad entre clases que no tienen relación. Esto se llama **domain gap**: el
> modelo fue entrenado con fotos muy estandarizadas (una hoja, fondo uniforme) y **no generaliza**
> bien a imágenes que se ven diferentes. Es una **limitación honesta** que tendríamos que resolver
> con datos más variados."

---

## 🏁 Cierre de la parte funcional
**🎤 Qué decir:**
> "En resumen: un Mamba implementado desde cero, entrenado localmente, llega a **78.5%** en 29
> clases de enfermedades de plantas. Es **muy confiable en su dominio** y muestra **límites claros**
> fuera de él. Todo el código y los resultados están en nuestro repositorio de GitHub."

**Narrativa en una frase:** *funciona en general (demo) → con alta confianza en su dominio
(clasificar Test) → pero tiene límites fuera de él (clasificar imagen rara).*

---
---

# PARTE 2 — Apéndice: el código a grandes rasgos

> Objetivo: entender **qué hace cada archivo y cada pieza** para responder preguntas, sin
> memorizar línea por línea.

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

- **PatchEmbed:** parte la imagen en parches de 16×16 y los convierte en una **secuencia** de
  vectores (tokens). Mamba, como los Transformers, trabaja sobre secuencias.
- **MambaBlock (el "S6", el corazón):**
  - `in_proj`: expande el vector y crea dos ramas: `x` (la señal) y `z` (una **compuerta**).
  - `Conv1d causal`: mezcla cada token con sus **vecinos** (contexto local).
  - **SSM selectivo:** una recurrencia de "espacio de estados" donde los parámetros
    **Δ, B, C se calculan a partir de la entrada** → el modelo **decide qué recordar y qué
    olvidar** en cada paso. Esta es la idea clave de Mamba.
  - **selective scan:** recorre la secuencia propagando un estado interno `h`. Es una
    recurrencia de coste **lineal** O(L), no cuadrático como la atención de los Transformers.
  - `y * SiLU(z)`: la compuerta controla cuánta señal del SSM pasa.
- **Bidireccional:** una imagen no tiene "orden" natural, así que escaneamos la secuencia
  **hacia adelante y hacia atrás** y sumamos (idea de *Vision Mamba / Vim*).
- **Mean Pooling + Linear:** se promedian los tokens y una capa lineal da las 29 clases.

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
