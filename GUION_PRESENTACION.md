# Guion de presentación — Mamba (SSM) para visión
### Tarea 3 · Equipo 2 · Procesamiento de Imágenes · Exposición 8/10 de junio

> **Cómo usar este guion:** cada bloque es una diapositiva. "🖥️ En pantalla" = qué poner en el slide (texto corto + figura). "🎤 Qué decir" = lo que explica quien presenta (no leer literal, es la idea). Duración objetivo: **12–15 min**. Reparto sugerido: 3 personas (Bloques A=Punto 1, B=Punto 2, C=cierre).

---

## Slide 1 — Portada
**🖥️ En pantalla**
- Título: **Mamba: modelos de espacio de estados (SSM) para visión**
- Subtítulo: Clasificación de enfermedades en plantas (PlantVillage)
- Equipo 2 · nombres · fecha · materia

**🎤 Qué decir**
> "Buenas, somos el Equipo 2. Nos tocó una de las *Nuevas arquitecturas*: **Mamba**, de 2023. Vamos a explicar cómo funciona por dentro y luego mostrar nuestra implementación clasificando enfermedades en hojas de plantas."

---

## Slide 2 — Agenda
**🖥️ En pantalla**
1. El problema y el dataset
2. ¿Por qué Mamba? (limitaciones de CNN y Transformers)
3. La idea: State Space Models (SSM)
4. De SSM a Mamba: el mecanismo **selectivo**
5. Arquitectura capa por capa **(Punto 1)**
6. Vision Mamba: Mamba en imágenes
7. Implementación y resultados **(Punto 2)**
8. Conclusiones

**🎤 Qué decir**
> "Primero el contexto, después la teoría de Mamba capa por capa, y cerramos con nuestra implementación y resultados."

---

# BLOQUE A — PUNTO 1 (40%): explicar la arquitectura

## Slide 3 — El problema y el dataset
**🖥️ En pantalla**
- **PlantVillage**: fotos de hojas, sanas y enfermas
- **29 clases** (Apple Scab, Tomato Late Blight, Corn Common Rust, …)
- Train **53.693** · Val **12.067** · Test **1.358**
- Una grilla de 6–8 imágenes de ejemplo (sacar del notebook)

**🎤 Qué decir**
> "El objetivo es clasificar la enfermedad de una hoja a partir de la foto. Son 29 categorías entre varios cultivos. Es un problema de **clasificación de imágenes**, ideal para comparar una arquitectura nueva contra las clásicas."

---

## Slide 4 — ¿Por qué una arquitectura nueva?
**🖥️ En pantalla** (tabla corta)
| Arquitectura | Fortaleza | Limitación |
|---|---|---|
| **CNN** (ResNet…) | Muy buen sesgo local, eficiente | Campo receptivo **limitado** (contexto global cuesta) |
| **Transformer / ViT** | Contexto **global** vía atención | Costo **O(L²)** en cómputo y memoria |
| **Mamba (SSM)** | Contexto global con costo **O(L)** | Recurrente (más difícil de paralelizar) |

**🎤 Qué decir**
> "Las CNN ven bien lo local pero les cuesta el contexto global. Los Transformers resuelven eso con atención, pero la atención crece **al cuadrado** con el tamaño de la secuencia: el doble de tokens = cuatro veces el costo. Mamba busca lo mejor de ambos: **contexto global pero con costo lineal**."

---

## Slide 5 — La idea base: State Space Model (SSM)
**🖥️ En pantalla**
- Sistema continuo (de teoría de control):
  $$ h'(t) = A\,h(t) + B\,x(t) \qquad y(t) = C\,h(t) $$
- `x` = entrada, `h` = **estado oculto** (memoria), `y` = salida
- Diagrama: x → [estado h] → y, con un lazo de realimentación en h

**🎤 Qué decir**
> "Un SSM viene de la ingeniería de control. La clave es el **estado oculto h**: una memoria que se va actualizando con cada entrada. `A` dice cómo evoluciona la memoria, `B` cómo entra la información nueva, y `C` cómo se lee la salida. Es básicamente una **RNN muy bien diseñada**."

---

## Slide 6 — Discretización (de lo continuo a lo discreto)
**🖥️ En pantalla**
- Como trabajamos con secuencias discretas (tokens), se discretiza con un paso **Δ**:
  $$ h_t = \bar{A}\,h_{t-1} + \bar{B}\,x_t \qquad y_t = C\,h_t $$
  $$ \bar{A} = \exp(\Delta A), \quad \bar{B} \approx \Delta B $$
- Δ ("delta") = "cuánto tiempo/peso doy a este paso"

**🎤 Qué decir**
> "Para usarlo en secuencias discretas se discretiza con un paso Δ. Esto convierte el sistema continuo en una **recurrencia**: el estado nuevo = una parte del estado viejo (Ā) más la entrada nueva (B̄x). Δ regula cuánta importancia tiene cada paso."

---

## Slide 7 — El problema de los SSM "fijos" (S4)
**🖥️ En pantalla**
- En S4, los parámetros `A, B, C, Δ` son **fijos** (iguales para todos los tokens) → sistema **lineal e invariante en el tiempo (LTI)**
- Ventaja: se puede calcular como una **convolución global** (rapidísimo)
- Problema: **no distingue** qué información es importante → no puede "filtrar" según el contenido

**🎤 Qué decir**
> "La versión anterior, S4, era muy eficiente porque los parámetros eran fijos y todo se podía hacer como una convolución. Pero al ser fijos, el modelo **trata igual a todos los tokens**: no puede decidir 'esto sí lo recuerdo, esto lo ignoro'. Ahí entra la innovación de Mamba."

---

## Slide 8 — La innovación de Mamba: mecanismo SELECTIVO (S6)
**🖥️ En pantalla** (idea central, resaltar)
- Mamba hace que **Δ, B y C dependan de la entrada**: $\Delta, B, C = f(x_t)$
- → El modelo **decide dinámicamente** qué recordar y qué olvidar en cada paso
- Costo: ya no es convolución → se resuelve con un **selective scan** (escaneo paralelo eficiente en GPU)

**🎤 Qué decir**
> "La gran idea de Mamba: hacer que los parámetros **dependan de la entrada**. Ahora, para cada token, el modelo calcula su propio Δ, B y C. Esto le da **atención implícita**: puede enfocarse en lo relevante e ignorar el ruido, como hace la atención del Transformer, pero **sin el costo cuadrático**. El precio es que ya no es una convolución, así que usan un algoritmo de *scan* optimizado para GPU."

---

## Slide 9 — El bloque Mamba, capa por capa (CLAVE del Punto 1)
**🖥️ En pantalla** (diagrama vertical del bloque)
```
        entrada x  (B, L, D)
            │
        in_proj  ──► se divide en  (x , z)     # expande canal + rama de compuerta
            │
        Conv1d causal + SiLU                    # mezcla LOCAL entre tokens vecinos
            │
        SSM selectivo (S6):
            Δ, B, C = Linear(x)                 # parámetros dependientes de la entrada
            A = -exp(A_log)                     # matriz de estado estable
            selective scan: h=Āh+B̄x ; y=Ch     # recurrencia GLOBAL, costo O(L)
            │
        y = y * SiLU(z)                         # COMPUERTA multiplicativa (gating)
            │
        out_proj                                # vuelve a dimensión D
            │
        salida  (B, L, D)
```

**🎤 Qué decir** (ir señalando cada capa)
> - "**in_proj**: una capa lineal que expande la dimensión y crea dos ramas: la señal `x` y una compuerta `z`."
> - "**Conv1d causal**: da contexto **local** — mezcla cada token con sus vecinos inmediatos. Mamba combina esto (local) con el SSM (global)."
> - "**SSM selectivo**: el corazón. Aquí se calculan Δ, B, C desde la entrada y se hace el *scan* que propaga la memoria por toda la secuencia."
> - "**A = -exp(A_log)**: se parametriza así para garantizar que el sistema sea **estable** (que la memoria no explote)."
> - "**compuerta y·SiLU(z)**: controla cuánta señal del SSM deja pasar, como un filtro aprendido."
> - "**out_proj**: regresa a la dimensión original. Todo el bloque va con conexión **residual** y normalización (RMSNorm)."

---

## Slide 10 — Vision Mamba: ¿cómo se usa Mamba en imágenes?
**🖥️ En pantalla**
- Mamba nació para **secuencias** (texto). Una imagen no es secuencia… ¿cómo?
1. **Patch Embedding**: partir la imagen en parches (ej. 16×16) → cada parche es un "token" (como en ViT)
2. **Positional Embedding**: añadir posición (Mamba no sabe dónde está cada parche)
3. **Escaneo bidireccional** (Vim): una imagen no tiene orden causal, así que se escanea **hacia adelante y hacia atrás**
4. (VMamba va más allá con un *cross-scan* 2D en 4 direcciones)

**🎤 Qué decir**
> "Mamba trabaja con secuencias, así que convertimos la imagen en una **secuencia de parches**, igual que ViT. Le agregamos información de posición. Y como una imagen no se 'lee' en un orden natural como el texto, escaneamos en **ambos sentidos** para que cada parche vea el contexto completo. Esa es la idea de *Vision Mamba (Vim)*."

---

## Slide 11 — Mamba vs Transformer vs CNN (resumen)
**🖥️ En pantalla** (tabla)
| Criterio | CNN | Transformer/ViT | **Mamba (SSM)** |
|---|---|---|---|
| Contexto | Local | Global | **Global** |
| Costo vs longitud L | O(L) | **O(L²)** | **O(L)** |
| Memoria | Baja | Alta | Baja |
| Selectividad de contenido | No | Sí (atención) | **Sí (Δ,B,C = f(x))** |
| Secuencias largas | Limitado | Costoso | **Ideal** |

**🎤 Qué decir**
> "Resumiendo: Mamba logra el contexto global del Transformer y la selectividad de la atención, pero con el **costo lineal** de una CNN. Por eso es tan prometedor para secuencias largas: video, imágenes de alta resolución, audio, ADN."

---

# BLOQUE B — PUNTO 2 (60%): implementación y resultados

## Slide 12 — Nuestra implementación: dos enfoques
**🖥️ En pantalla**
- **Parte A — Vision Mamba desde cero (PyTorch puro):** implementamos el bloque S6 y el *selective scan* a mano → para **entender cada capa**.
- **Parte B — Transfer Learning con MambaVision (NVIDIA):** modelo Mamba **preentrenado en ImageNet**, le cambiamos la cabeza a 29 clases y hacemos **fine-tuning** → lo que pidió el profe.

**🎤 Qué decir**
> "Hicimos dos cosas. Primero implementamos Vision Mamba **desde cero** para demostrar que entendemos la arquitectura. Y segundo, aplicamos **transfer learning**: tomamos un Mamba ya entrenado, le reemplazamos la última capa para nuestras 29 clases y lo re-entrenamos. Así comparamos los dos enfoques."

---

## Slide 13 — Transfer Learning, ¿qué es?
**🖥️ En pantalla** (diagrama)
```
ImageNet (1000 clases)            PlantVillage (29 clases)
┌────────────────┐                ┌────────────────┐
│ Backbone Mamba │  reutilizamos  │ Backbone Mamba │ (pesos preentrenados)
├────────────────┤  ───────────►  ├────────────────┤
│ Cabeza → 1000  │  reemplazamos  │ Cabeza →  29   │ (nueva, se entrena)
└────────────────┘                └────────────────┘
```
- El backbone ya sabe extraer características visuales → solo re-entrenamos la cabeza (rápido) y luego un *fine-tuning* suave del resto.

**🎤 Qué decir**
> "La idea del transfer learning: el modelo ya aprendió a 'ver' con millones de imágenes de ImageNet. No empezamos de cero; **reaprovechamos** ese conocimiento y solo le enseñamos las clases nuevas. Es más rápido y da mejor resultado."

---

## Slide 14 — Pipeline de datos
**🖥️ En pantalla**
- `ImageFolder` (cada carpeta = una clase)
- **Aumentos** en train: RandomResizedCrop, flip, rotación, color jitter
- Normalización ImageNet · imágenes a 160px (Parte A) / 224px (Parte B)
- Mixed precision (AMP) para acelerar en GPU T4

**🎤 Qué decir**
> "Cargamos las imágenes por carpetas, aplicamos aumentos de datos para que el modelo generalice mejor, y usamos precisión mixta para entrenar más rápido en la GPU gratuita de Colab."

---

## Slide 15 — Resultados Parte A (from scratch)
**🖥️ En pantalla**
- Curvas de **loss** y **accuracy** (train/val) — captura del notebook
- Accuracy en **Test: ___ %** (rellenar al correr)
- Matriz de confusión (captura)

**🎤 Qué decir**
> "Entrenando desde cero unas pocas épocas, llegamos a __ % en test. En la matriz de confusión se ve que la mayoría de clases se predicen bien; las confusiones suelen ser entre enfermedades visualmente parecidas del mismo cultivo."

---

## Slide 16 — Resultados Parte B (transfer learning)
**🖥️ En pantalla**
- Accuracy en **Test: ___ %** (rellenar)
- Algunas predicciones de ejemplo (verde=acierto / rojo=error)

**🎤 Qué decir**
> "Con transfer learning, partiendo del modelo preentrenado, en **menos épocas** alcanzamos __ %, bastante más alto. Confirma lo que decía el profe: re-entrenar un modelo del estado del arte rinde mucho mejor que empezar de cero."

---

## Slide 17 — Comparación
**🖥️ En pantalla**
- Gráfica de barras: from-scratch vs transfer learning (captura del notebook)
- Idea: **menos datos/tiempo + mejor accuracy** con transfer learning

**🎤 Qué decir**
> "Lado a lado: el transfer learning gana en accuracy y converge más rápido. El from-scratch, en cambio, nos sirvió para **entender la arquitectura por dentro**, que era el objetivo del Punto 1."

---

# BLOQUE C — Cierre

## Slide 18 — Conclusiones
**🖥️ En pantalla**
- Mamba = contexto **global** con costo **lineal** O(L) gracias al SSM **selectivo**
- Lo adaptamos a imágenes (parches + escaneo bidireccional)
- Implementación desde cero (entender) + transfer learning (mejor desempeño)
- Resultado: clasificación de 29 enfermedades de plantas con buena accuracy

**🎤 Qué decir**
> "Mamba es una alternativa eficiente a los Transformers: misma capacidad de contexto global, pero escala linealmente. Lo implementamos de las dos formas y funcionó bien en un problema real de visión."

---

## Slide 19 — Limitaciones y trabajo futuro
**🖥️ En pantalla**
- El *selective scan* en PyTorch puro es lento (sin kernels CUDA de `mamba-ssm`)
- Modelos/épocas pequeños por la GPU gratis → hay margen para subir accuracy
- Futuro: probar VMamba (scan 2D), más épocas, más resolución

**🎤 Qué decir**
> "Nuestra versión casera es lenta porque no usamos los kernels optimizados; en producción se usarían. Con más cómputo y épocas, la accuracy subiría más."

---

## Slide 20 — Cierre / preguntas
**🖥️ En pantalla**
- "¡Gracias!" + nombres del equipo
- (Opcional) link al notebook / repo

**🎤 Qué decir**
> "Eso es todo, ¿preguntas?"

---

## 📌 Anexo — posibles preguntas del profe (prepárense)
- **¿Por qué Δ, B, C dependientes de la entrada mejoran?** → Dan **selectividad**: el modelo filtra contenido relevante, como la atención, pero sin O(L²).
- **¿Diferencia S4 vs S6 (Mamba)?** → S4 tiene parámetros fijos (LTI, es convolución); S6 los hace input-dependent (selectivo, requiere scan).
- **¿Por qué A = −exp(A_log)?** → Para garantizar autovalores negativos → sistema **estable**, la memoria no explota.
- **¿Por qué bidireccional en imágenes?** → No hay orden causal natural; escanear en ambos sentidos da contexto completo a cada parche.
- **¿Mamba reemplaza al Transformer?** → No necesariamente; es muy fuerte en **secuencias largas** y eficiencia; existen híbridos (MambaVision combina Mamba + atención).
- **¿Qué es el `selective scan`?** → El algoritmo que calcula la recurrencia h_t en paralelo en GPU, ya que con parámetros variables no se puede usar la FFT/convolución de S4.
