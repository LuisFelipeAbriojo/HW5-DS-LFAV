# De la voz al registro — Entrevistas y llamadas de venta

Proyecto integrador de ASR: una entrevista grabada se convierte en una **ficha de candidato** y una llamada de venta
en un **registro de pedido que se verifica a sí mismo**, con el mismo motor.

- **Documento de pipeline (Notion, público):** `<<PEGA AQUÍ EL ENLACE>>` — fuente versionada en [docs/pipeline.md](docs/pipeline.md)
- **Video (≤ 8 min):** `<<PEGA AQUÍ EL ENLACE>>`
- **Dominio de la Tarea 2:** restaurante (pollería). Carta, tamaños y promociones en [config.yaml](config.yaml).

> **Datos inventados.** Todo lo que se dice en los audios es ficticio: no hay salarios, direcciones, empresas ni datos
> personales reales de nadie. Los nombres de los personajes ("Mateo", "Andrea", "La Brasa Dorada", "Andina Retail") son inventados.
>
> **Consentimiento.** Todas las personas que aparecen en las grabaciones aceptaron participar y que su voz se use
> en este proyecto. No se nombra a nadie que no haya aceptado ser nombrado. `<<confirma y, si quieres, nombra a tu(s) compañero(s)>>`

---

## 1. Instalación y ejecución

```bash
python -m venv .venv && .venv\Scripts\activate          # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
cp .env.example .env                                      # solo si usarás Claude en vez de Ollama: ANTHROPIC_API_KEY
```

| Qué | Comando | ¿GPU? |
|---|---|---|
| Verdad de campo desde los guiones | `python scripts/build_ground_truth.py` | no |
| Benchmark ASR (tamaños, precisión, CPU/GPU, glosario) | `python scripts/benchmark_asr.py` | **sí** (Colab) |
| Transcribir el corpus a caché | `python scripts/transcribe_corpus.py --task interview` | **sí** |
| Construir la escalera L1–L4 | `python scripts/build_ladder.py --task order` | no |
| Transcribir la escalera | `python scripts/transcribe_corpus.py --task order --levels L0 L1 L2 L3 L4` | **sí** |
| Métricas de ambas capas (desde caché) | `python scripts/evaluate.py` | no |
| Tablas Markdown para este README | `python scripts/make_tables.py` | no |
| App | `streamlit run app.py` | no |
| Pruebas | `python -m pytest -q` | no |
| **El motor no importa UI** | `python scripts/check_engine_imports.py` | no |

Todo lo que necesita GPU está en [notebooks/colab_gpu.ipynb](notebooks/colab_gpu.ipynb) (Colab gratis, T4).
La app y la evaluación **solo leen** `cache/` y `results/`: nunca transcriben el corpus al iniciar.

### Audio y pesos de modelo
Los audios (`data/audio/`) y los pesos no se versionan (`.gitignore`). Los pesos se descargan solos la primera vez
(Hugging Face, `Systran/faster-whisper-*`). Para reproducir desde audio, solicita los archivos a `<<tu correo>>`;
para reproducir las métricas **no hacen falta**: las transcripciones cacheadas están en `cache/transcripts/`.

---

## 2. Arquitectura: un motor, dos esquemas

```
vozregistro/
  engine.py        process_audio(ruta, schema, method) -> dict   ← única función que llama la app
  asr.py           faster-whisper + dispositivo leído del modelo + VRAM medida (NVML)
  audio.py         decodificación PyAV
  numbers.py       capa de normalización numérica (palabras -> dígitos)
  textnorm.py      regla de normalización para WER/CER y verificación
  verify.py        regla de verificación literal
  extract_llm.py   ruta LLM (Ollama local por defecto, o Claude; JSON schema estricto, respuestas cacheadas)
  schemas/
    interview.py   6 campos: reglas, schema JSON, verificación, puntuación
    order.py       pedido: reglas, schema JSON, verificación, puntuación
  validate.py      validación aritmética + diagnóstico de causa
  ladder.py        escalera de degradación
  metrics.py       WER, CER, Wilson, bootstrap, F1 macro
```

`process_audio` devuelve siempre: `transcript`, `fields`, `verified`, `evidence`, `validation` (pedidos),
`device`, `compute_type`, `model`, `elapsed_s`, `error`. El esquema es un **módulo enchufable** con la misma interfaz
(`extract_rules`, `llm_schema`, `from_llm`, `verify`, `score`); `engine.py` no sabe nada de candidatos ni de pollos.
Cambiar el modelo ASR o el LLM es cambiar una línea de `config.yaml`, y afecta a las dos tareas a la vez.

**Comando de verificación de que el motor no importa UI:** `python scripts/check_engine_imports.py`
(recorre el AST de `vozregistro/` buscando `streamlit`, `gradio`, `flask`, `plotly`, etc.; corre también en CI).

**Configuración fuera del código:** rutas, modelos, umbrales, glosarios, carta, promociones, prompts del LLM y
parámetros de la escalera viven en `config.yaml`. Credenciales solo en `.env` (ver `.env.example`, solo nombres).

---

## 3. Tarea 1 — Entrevistas

### Corpus y verdad de campo
Guiones en [data/scripts/interview/](data/scripts/interview/) (escritos desde cero, no el del apéndice). La verdad
de campo **no costó nada**: el guion se escribió antes de grabar, así que la transcripción exacta y los valores
correctos ya existían. `build_ground_truth.py` genera [data/ground_truth/interview.csv](data/ground_truth/interview.csv),
las referencias en `data/ground_truth/reference/` y la distribución de campos en `distribution.json`.
Si al grabar alguien se desvía del guion, se corrige el `.yaml` (la referencia debe ser lo que se dijo).

**Cómo se varió la posición y la forma de cada campo:**

| Clip | Orden de campos | Trampas plantadas |
|---|---|---|
| E01 | exp → puesto → tecnologías → inglés → disponibilidad → **salario al final** | "hace como cinco años" antes de "cuatro"; salario actual 5800 y un "seis mil" a medias antes de 6500 |
| E02 | **salario primero** → inglés → exp → puesto → tecnologías → disponibilidad | inglés como certificación (C1); "dos años y medio" vs "como tres"; disponibilidad categórica |
| E03 | **puesto primero** → disponibilidad → tecnologías → exp → salario → inglés | puesto anterior mencionado; **tecnología negada** (Azure); 7 junto a 3 y 4; salario en USD; fecha absoluta |
| E04 | **tecnologías primero** → inglés → salario → exp → puesto → disponibilidad | **rango** salarial antes de la cifra; "un año en marketing" que no cuenta |
| E05 | exp+puesto en una frase → salario → tecnologías → disponibilidad → inglés | "año y medio" junto al total; cifra no redonda (9350); preaviso en días; B2 |

### Cómo grabar
- Tú apareces en **todos** los clips (como A o como B); alterna roles con tu compañero. Cada uno arma su propio set.
- Celular, nota de voz normal. Nombra el archivo como el guion: `data/audio/interview/E01.m4a`, `data/audio/order/P01.m4a`.
- Habla natural, con pausas; 2–4 min por clip (leídos por TTS duran 1:40–2:55, la voz humana suele ir más lenta).
- Si improvisas o te equivocas, **corrige el `.yaml`** para que la referencia sea lo dicho y vuelve a correr `build_ground_truth.py`.
- Graba además ~1 min de ambiente real (cafetería, calle) en `data/noise/ambiente.wav` para el nivel L3.

### Normalización (WER/CER y verificación)
Misma regla para referencia e hipótesis: minúsculas → sin tildes (ñ→n) → números en palabras a dígitos → sin
puntuación (se conserva el punto decimal y `%`) → espacios colapsados ([textnorm.py](vozregistro/textnorm.py)).

### Capa de normalización numérica ([numbers.py](vozregistro/numbers.py))
"seis mil quinientos" en la transcripción y 6500 en el registro fallan la verificación literal aunque todo esté
bien. Reglas: R1 "por ciento"→`%`; R2 formatos del ASR (`6.500`→6500, `39,90`→39.90); R3 gramática de numerales
(centenas→decenas→"y"→unidades; "mil"/"millón" multiplican; cuando una palabra no puede continuar el número empieza
otro: "treinta y nueve noventa"→`39 90`); R4 "N con M"→N.M; R5 precio peruano "cuatro cincuenta"→4.50 (salvo que
sigan unidades de tiempo); R6 "y medio"→.5; R7 "un/una"→1. Cubierto por `tests/test_numbers.py`.

### Regla de verificación ([verify.py](vozregistro/verify.py))
Cada campo trae un **valor** y una **evidencia** (el fragmento que lo sustenta). El campo es `verificado` solo si la
evidencia aparece literalmente en la transcripción normalizada **y** la evidencia se normaliza al mismo valor; si no,
`sin_verificar`. No hay coincidencia difusa. Un LLM que "corrige" 6600 por 6500 sin que nadie lo dijera queda
`sin_verificar` aunque acierte por casualidad (`tests/test_pipeline.py::test_verification_rejects_corrected_number`).
`evaluate.py` reporta además una ablación: cuántos campos numéricos se verificarían **sin** la capa numérica.

### Ruta LLM sin API key
Por defecto la extracción LLM usa **Ollama** con `qwen2.5:7b-instruct`: gratis, sin credenciales, y el texto no sale
de la máquina (el mismo argumento de la tabla del problema contra Otter/Fireflies). El esquema JSON se pasa en el
parámetro `format`, así que la salida está restringida por gramática a ese esquema. En Colab corre en la misma T4
después de liberar Whisper (ver notebook). Localmente: instala Ollama, `ollama pull qwen2.5:7b-instruct` y
`python scripts/evaluate.py`. Para usar Claude: `provider: anthropic` en `config.yaml` y la key en `.env`.

### Glosario: ¿`hotwords` o `initial_prompt`?
La documentación pública es inconsistente. Se resolvió **inspeccionando la versión instalada**
(`asr.glossary_support()` lee la firma de `WhisperModel.transcribe`): faster-whisper 1.2.1 acepta `hotwords` y
`initial_prompt`. `hotwords` no tiene efecto si se pasa `prefix`. El benchmark mide ambos contra "sin glosario" con
diferencia **pareada** por clip e IC bootstrap.

### Prueba de que se usó la GPU
El dispositivo **no** se toma de `config.yaml`: se lee de `model.model.device` (objeto CTranslate2 ya cargado) y se
respalda con la VRAM medida por NVML en un hilo durante carga + inferencia (`peak_vram_mb`, `delta_vram_mb`,
memoria del proceso). Un pipeline que cayó a CPU en silencio aparece con `device=cpu` y VRAM nula.

### Resultados
<!-- Pega aquí la salida de: python scripts/make_tables.py -->
`<<pendiente: correr notebooks/colab_gpu.ipynb y luego python scripts/make_tables.py>>`

**¿Se puede saber si el registro sirve mirando solo el WER?** `evaluate.py` busca en tus datos el clip con WER más
alto y registro perfecto, y el de WER más bajo con registro roto (`results/interview_eval.json → wer_vs_registro`,
gráfico en la pestaña 3 de la app). La asimetría esperada: un campo categórico (`nivel_ingles`) sobrevive a muchos
errores porque basta oír una palabra clave; una cifra (`pretension_salarial`) se rompe con un solo número mal oído
aunque el WER global sea bajo. Quien reporta solo WER no sabe si los datos sirven. `<<redacta con tus dos clips>>`

---

## 4. Tarea 2 — Llamadas de venta (restaurante)

### Corpus con trampas
Guiones en [data/scripts/order/](data/scripts/order/); tabla en [data/ground_truth/order.csv](data/ground_truth/order.csv).

| Llamada | Autocorrección | Total dicho erróneo | Dirección con dígitos | Detalle |
|---|---|---|---|---|
| P01 | ✔ (2 gaseosas → 1) | — | ✔ | total correcto 93.80 |
| P02 | — | ✔ dice 121, es 111 | ✔ | olvida restar el combo familiar |
| P03 | ✔ (3 medios → 2) | ✔ dice 78.48, es 75.48 | ✔ | descuento porcentual 15% |
| P04 | ✔ (cambia un tallarín por lomo) | — | ✔ (interior 4B) | gaseosa de regalo |
| P05 | ✔ (4 cuartos → 3) | ✔ dice 89.60, es 88.60 | ✔ (manzana F lote 12) | cupón −5 |
| P06 | ✔ cantidad, tamaño y método de pago | ✔ dice 191.70, es 190.70 | ✔ (+ dígitos distractores en la referencia) | cumpleaños familiar, combo −10 |

### Escalera de degradación ([ladder.py](vozregistro/ladder.py), reproducible con semillas)
| Nivel | Simula | Transformación |
|---|---|---|
| L0 | tal cual | decodificado a 16 kHz mono |
| L1 | nota de voz WhatsApp | Opus 16 kbit/s mono (libopus vía PyAV) |
| L2 | llamada telefónica | Butterworth orden 6, 300–3400 Hz, remuestreo a 8 kHz |
| L3 | ambiente ruidoso | ruido ambiente a **5 dB SNR** + códec de L1 |
| L4 | endurecimiento | ruido a 0 dB SNR + banda telefónica (se reporta aparte) |

**Por qué 5 dB:** un restaurante o una calle concurrida queda típicamente entre 0 y 10 dB de SNR respecto a la voz
al celular; 5 dB es el punto medio realista y deja L4 (0 dB) como prueba de estrés. **Cómo se mide:** VAD por energía
(tramas de 30 ms sobre el percentil 40) → potencia de la voz solo en tramas activas vs potencia del ruido escalado en
esas mismas tramas; la SNR medida se guarda por clip en `results/ladder_build_real_order.json`. El ruido se mezcla
**antes** del códec (el micrófono lo capta antes de comprimir). Si `data/noise/ambiente.wav` no existe se usa ruido
rosa con semilla y se declara (`noise_source`). `<<graba 1 min de ambiente real y colócalo ahí>>`
Si el registro sigue perfecto en L3, la escalera no degrada lo suficiente: por eso existe L4.

### Validación aritmética y diagnóstico ([validate.py](vozregistro/validate.py))
Σ cantidad × precio unitario − promoción = total dicho (± S/ 0.05). Cuando no cuadra, el sistema encontró un problema
**sin verdad de campo**. Regla de diagnóstico:
- **error_asr** si (a) algún precio unitario no coincide con la carta, (b) alguna palabra numérica tras la última
  mención de "total" tiene confianza ASR < 0.60, o (c) recalculando con precios de carta el total dicho sí cuadra;
- **error_hablante** si todo es coherente con la carta y se oyó con confianza, y aun así no cuadra;
- **incompleto** si faltan ítems, precios o total.

Para medir qué tan seguido acierta, la "causa real" de cada alarma se define con la verdad de campo: si el registro
extraído coincide con el guion y la llamada tenía total erróneo plantado → error del hablante; si no → error de
ASR/extracción. Se reporta por nivel: atrapados, perdidos, falsas alarmas y aciertos del diagnóstico.

### Resultados
<!-- Pega aquí la salida de: python scripts/make_tables.py (sección Tarea 2) -->
`<<pendiente>>`

---

## 5. Innovación incluida
- **Puerta de regresión en GitHub Actions** ([.github/workflows/regression.yml](.github/workflows/regression.yml)):
  recalcula métricas desde la caché y falla bajo los umbrales de `config.yaml`. Los runners no tienen GPU: el
  workflow nunca transcribe (si lo hiciera, caería a CPU y sería ~40× más lento o se quedaría sin tiempo).
- **Eje de acento sintético** (opcional): `python scripts/synthesize_tts.py --task interview --locale es-MX`
  (es-PE, es-MX, es-ES, es-AR, es-CO). Son voces sintéticas: el resultado describe cómo el modelo trata esas voces,
  no acentos humanos reales. Grupo `synthetic`, siempre separado de las grabaciones.

## 6. Herramientas de IA declaradas
| Herramienta | Para qué |
|---|---|
| Claude Code (Claude Opus 5.5) | Andamiaje del repositorio, motor, extractores, métricas, escalera, app y borradores de guiones; yo revisé, ejecuté y ajusté cada paso |
| Ollama + `qwen2.5:7b-instruct` (modelo abierto) | Ruta de extracción LLM con JSON estructurado dentro del pipeline, gratis y local (en Colab, misma GPU) |
| Claude API (`claude-opus-5-5`) | Proveedor alternativo de la ruta LLM (`provider: anthropic`); `<<indica si lo usaste>>` |
| Servidores MCP conectados en Claude Code | `<<ninguno / lista los que uses, con una línea de qué hizo cada uno>>` |

## 7. Limitaciones y lo que la muestra no soporta
- **n pequeño:** 5 entrevistas × 6 campos = 30 observaciones; 5 llamadas × 5 campos = 25 por nivel. Los IC de Wilson
  son anchos: diferencias de ±20 puntos entre métodos pueden ser ruido. Las comparaciones de glosario son pareadas y
  aun así con 5 clips el IC suele incluir 0. No se afirma un efecto cuyo IC incluya 0.
- **Las reglas se escribieron mirando estos guiones.** Su exactitud sobre la referencia (100%) es un techo optimista;
  sobre audio real y sobre guiones nuevos bajará. La ruta LLM no tiene ese sesgo pero tiene otros: un modelo abierto de 7B copia peor la evidencia literal y sigue peor las autocorrecciones que un modelo grande.
- **Hablantes:** dos o pocas voces, un acento (Lima), un celular. No describe otros acentos, edades ni micrófonos.
- **Sin diarización:** las reglas no saben quién habla; un número dicho por el entrevistador puede confundirse con el
  del candidato (se mitiga con señales de contexto, no se resuelve).
- **Glosario cerrado:** la ruta de reglas solo encuentra tecnologías que están en el glosario.
- **El diagnóstico** se calibró con errores plantados y con degradación artificial; errores reales de otros tipos
  (p. ej. el cliente cambia de idea después del total) no están representados.

## 8. Lo que este sistema nunca debe usarse para
- **Decidir contrataciones de forma automática** ni puntuar candidatos: la ficha es una ayuda de registro, no un juicio.
- Procesar grabaciones de personas que **no consintieron** ser grabadas.
- Cobrar o facturar sin revisión humana cuando un campo está `sin_verificar` o la validación no cuadra.
- Inferir atributos personales (acento, origen, género, edad) a partir de la voz.
