# Guion — Bloque de apertura (estética terminal)
### Tarea 3 · Equipo 2 · Procesamiento de Imágenes · Exposición 8/10 de junio

> **Cómo usarlo:** "🖥️ En pantalla" = lo que ya está / pondrías en el slide. "🎤 Qué decir" = lo que narra quien presenta (idea, no leer literal). Tono relajado pero técnico. Tiempo total de estos 4 slides: **~3–4 min**.

---

## Slide 0 — Bienvenida / `BOOT`
**🖥️ En pantalla** (sugerencia, mismo estilo verde-terminal)
```
SESSION: INITIALIZATION ............ LOG_01
> Cargando arquitectura: MAMBA (2023)
> Equipo 2 · Procesamiento de Imágenes
> Status: READY ▮
```
- Título grande: **MAMBA** · subtítulo: *Modelos de Espacio de Estados para visión*
- Nombres del equipo · fecha (8/10 jun)

**🎤 Qué decir**
> "Buenas a todos. Somos el **Equipo 2**. Nos tocó una de las arquitecturas nuevas: **Mamba**, de 2023. En los próximos minutos vamos a 'arrancar el sistema': primero qué es Mamba y por qué apareció, después sus tres pilares, y luego abrimos el capó para ver la capa que hace toda la magia — el SSM. Al final lo mostramos funcionando clasificando enfermedades en hojas de plantas. Arrancamos."

*(El truco de presentación: conecta el "Status: READY" con "arrancamos" — la estética de consola te da un hilo narrativo gratis.)*

---

## Slide 1 — `LOG_02` · ¿Qué es Mamba?
**🖥️ En pantalla**
> Mamba es una nueva arquitectura de red neuronal basada en Modelos de Espacio de Estados (SSM). A diferencia de los Transformers, logra complejidad **lineal O(N)** en lugar de cuadrática, permitiendo procesar secuencias de **un millón de tokens** con eficiencia computacional extrema.

**🎤 Qué decir**
> "¿Qué es Mamba en una frase? Es una red basada en **modelos de espacio de estados**, una idea que viene de la teoría de control. ¿Por qué nos importa? Por **esto** —señalar el O(N)—. El Transformer, que es hoy el rey, tiene un problema: su mecanismo de atención crece **al cuadrado** con la longitud de la secuencia. El doble de datos cuesta **cuatro veces** más. Mamba logra lo mismo —mirar todo el contexto— pero con costo **lineal**: el doble de datos, el doble de costo, y ya. Por eso puede tragarse secuencias de **un millón de tokens** donde un Transformer se ahoga. Esa es la promesa; ahora veamos cómo lo consigue."

*(Gancho clave: enfatiza el contraste O(N²) vs O(N) con las manos — es el "por qué existe Mamba".)*

---

## Slide 2 — Pilares Fundamentales
**🖥️ En pantalla** (3 tarjetas)
> **01 Selección** — filtrar información relevante según el contenido de la entrada.
> **02 Discretización** — convertir sistemas continuos a representaciones discretas.
> **03 Hardware-Aware** — optimización para GPU que evita materializar estados intermedios.

**🎤 Qué decir** (ir tarjeta por tarjeta)
> "Mamba se sostiene sobre **tres pilares**.
> **Uno, Selección** —el más importante—. Los modelos anteriores trataban a todos los datos por igual. Mamba **decide en cada paso** qué recuerda y qué ignora, según el contenido. Es como una atención implícita: filtra el ruido y se queda con lo que importa.
> **Dos, Discretización.** El modelo nace en tiempo *continuo*, ecuaciones tipo física. Como las computadoras y nuestras secuencias son **discretas** —tokens, parches—, hay que convertirlo a pasos discretos. Ese es el rol del parámetro **Δ**, que veremos enseguida.
> **Tres, Hardware-Aware.** Y esto es ingeniería pura: el algoritmo está **diseñado pensando en la GPU**. Evita guardar estados intermedios en la memoria lenta, y por eso, aunque por dentro es recurrente, en la práctica vuela.
> Selección, discretización y diseño consciente del hardware: esos tres juntos son lo que hace a Mamba viable."

*(Reparto: si presentan entre varios, este slide se puede repartir una tarjeta por persona.)*

---

## Slide 3 — `LOG_04` · Capa SSM Estructurada
**🖥️ En pantalla** (matrices A, B, C, D + texto)
> La base de Mamba es el SSM clásico definido por las matrices **(A, B, C, D)**. Mapea una secuencia de entrada x(t) a una salida y(t) a través de un **estado latente h(t)**. Captura dependencias de largo alcance sin el costo de la atención global.

**🎤 Qué decir**
> "Abramos el capó. El corazón de Mamba es esta capa, el **SSM**, y se define con cuatro matrices: **A, B, C y D**. La pieza central es el **estado latente h(t)** —esa de en medio—: piénsenlo como una **memoria** que el modelo va arrastrando a lo largo de la secuencia.
> El flujo es simple: entra **x(t)**, ese estado **h** se actualiza, y de ahí sale **y(t)**. Cada matriz tiene su papel: **A** controla cómo evoluciona la memoria de un paso al siguiente, **B** cómo entra la información nueva, **C** cómo se lee la salida desde el estado, y **D** es una conexión directa de la entrada a la salida.
> ¿Por qué es tan potente? Porque esa memoria **h** viaja por toda la secuencia, así que un token del final puede 'acordarse' de algo del principio —**dependencias de largo alcance**— pero **sin** comparar todos contra todos como hace la atención. Ahí está el ahorro: contexto global, costo lineal."

*(Pista: si la diapositiva siguiente es el mecanismo selectivo S6, cierra con — "y la versión de Mamba hace que B y C dependan de la entrada… pero eso es el siguiente log".)*

---

## 🔗 Hilo narrativo entre los 4 slides
1. **Bienvenida** → "arrancamos el sistema"
2. **Qué es** → *el porqué*: O(N) vs O(N²)
3. **Pilares** → *el cómo, a alto nivel*: selección + discretización + hardware
4. **Capa SSM** → *el cómo, por dentro*: A, B, C, D y el estado h

Cada slide responde la pregunta que deja el anterior. Eso es lo que hace que fluya.
