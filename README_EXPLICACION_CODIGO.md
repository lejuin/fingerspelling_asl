# README_EXPLICACION_CODIGO

Documento de apoyo para entender el proyecto sin tecnicismos y dejar por escrito los problemas detectados.

Fecha: 2026-03-06

## 1) Que intenta hacer este proyecto

El objetivo es leer el movimiento de la mano derecha y convertirlo en letras (fingerspelling ASL).
Con esas letras, se puede formar una palabra.

Hay dos usos principales:

- Entrenar un modelo con datos guardados (archivos parquet).
- Probar un modelo ya entrenado con la webcam.

## 2) Como esta organizado el codigo

Carpetas principales:

- `src/`
  - `train.py`: entrenamiento principal.
  - `quick_infer.py`: prueba rapida sobre datos del dataset.
  - `realtime_webcam.py`: solo deteccion de mano y visualizacion.
  - `realtime_webcam_infer.py`: demo webcam con prediccion de letras.
  - `model_loader.py`: cargador que detecta que tipo de modelo hay dentro del checkpoint.
  - `data/`
    - `dataset.py`: lectura de parquets y construccion de lotes para entrenar/evaluar.
    - `vocab.py`: utilidades de vocabulario.
  - `models/`
    - `embedded_rnn.py`: modelo base simple.
    - `tcn_bilstm.py`: arquitectura usada por el run final grande.
  - `utils/`
    - `metrics.py`: CER/WER/ExactMatch/AvgEditDist.
    - `ctc_decode.py`: decode CTC basico.
- `docs/`: guias de troubleshooting.
- `artifacts/`: checkpoints, logs y modelos auxiliares.

## 3) Flujo simple de extremo a extremo

1. Se leen archivos de landmarks de mano derecha.
2. Se emparejan con su texto objetivo.
3. Se entrena un modelo para convertir secuencia de movimientos en secuencia de letras.
4. Se mide calidad (CER/WER/etc).
5. Se guarda un checkpoint.
6. Se usa ese checkpoint en inferencia (dataset o webcam).

## 4) Que hace cada pieza importante

### `src/data/dataset.py`

- Busca columnas de mano derecha en parquet.
- Saca secuencia por `sequence_id`.
- Rellena o recorta a `max_frames`.
- Convierte NaN a 0.
- Devuelve:
  - entrada `X`
  - objetivo `Y`
  - longitud de entrada
  - longitud objetivo

### `src/train.py`

- Carga CSV y vocabulario.
- Parte train/val por participante.
- Crea DataLoaders.
- Entrena con CTC loss.
- Calcula metricas de validacion.
- Guarda checkpoint por epoca.
- Si W&B esta activo, loggea metricas y 5 ejemplos GT/PRED.

### `src/model_loader.py`

- Abre un checkpoint.
- Detecta que arquitectura contiene.
- Si es modelo base simple: usa `EmbeddedRNN`.
- Si es modelo grande de comparativa: usa `TCNBiRNN`.

### `src/realtime_webcam_infer.py`

- Detecta mano en webcam (MediaPipe).
- Convierte landmarks a vector numerico.
- Ajusta el vector al tamano esperado por el checkpoint.
- Predice letra en tiempo real.
- Muestra indicadores (top3, confianza, voto temporal, etc).
- Tiene modo guiado con `ESPACIO` para capturar letra puntual.

## 5) Auditoria de inconsistencias (priorizada por gravedad)

## Critica

1. Desalineacion fuerte entre el run final y el `train.py` actual del repo.
- Evidencia: el checkpoint final `archcmp2_tcn_bilstm_full_20260303_best.pt` trae configuracion con 62 claves.
- El parser actual de `train.py` solo expone 22 opciones.
- Hay 40 opciones del run final que no existen en el `train.py` actual (`arch`, `use_delta_features`, `augment_*`, `early_stopping_*`, `weight_decay`, `num_workers`, etc).
- Impacto: no se puede reproducir fielmente el run final desde este repo.

2. `train.py` actual entrena solo `EmbeddedRNN` con `input_dim=63` fijo.
- El checkpoint final esperado por demo usa `TCN_BiLSTM` y `input_dim=126`.
- Impacto: si alguien vuelve a entrenar con el script actual, obtendra un modelo distinto al que se esta intentando usar en demo.

3. Riesgo de conclusion incorrecta del rendimiento por falta de trazabilidad exacta de codigo.
- El run final parece hecho con una version mas avanzada del entrenamiento que no estaba reflejada en repo.
- Impacto: se comparan peras con manzanas entre “lo entrenado” y “lo que el repo enseña”.

## Alta

1. Diferencias de preprocesado entre caminos de entrenamiento e inferencia local.
- En webcam se aplica centrado/escala y opcion delta para 126.
- En `dataset.py` actual no se aplica ese mismo pipeline de forma explicita para entrenamiento.
- Impacto: el modelo puede ver distribuciones distintas en train vs inferencia.

2. Calidad de demo webcam limitada por objetivo del modelo.
- El modelo fue entrenado para secuencias de fingerspelling (frases/palabras), no para clasificar una pose estatica de una sola letra.
- Impacto: al enseñar una letra fija puede dar resultados inestables o ambiguos.

3. Señal de configuracion inconsistente en checkpoint.
- En la config guardada aparece `rnn_type='rnn'`, pero los pesos cargados corresponden a estructura tipo LSTM (por dimensiones de puertas).
- Impacto: confunde al depurar y documentar arquitectura real.

## Media

1. WER puede ser poco informativo cuando el setup esta en `letters_only`.
2. Falta de tests automatizados para validar compatibilidad de features entre:
- `dataset.py`
- `quick_infer.py`
- `realtime_webcam_infer.py`
3. Falta una “ficha de experimento” por run (codigo exacto, commit exacto, datos exactos, parametros exactos).

## 6) Por que pudo salir “tan mal” el ultimo run y la demo webcam

Posibles causas mas probables, en orden:

1. El resultado de validacion (CER ~0.51) probablemente pertenece a un pipeline distinto del `train.py` actual.
2. El modelo de secuencia no necesariamente es bueno para “una sola letra estatica en webcam”.
3. La webcam introduce ruido real (iluminacion, camara, angulo, mano parcial, FPS, jitter).
4. El criterio de decision online (umbrales/voto/estado) necesita calibracion por usuario/entorno.

Nota: esto no significa que el modelo “no sirva”; significa que la forma de evaluarlo en vivo puede no estar alineada con como fue entrenado.

## 7) Pruebas recomendadas (plan concreto)

## Bloque A (obligatorio, alta prioridad)

1. Congelar reproducibilidad del run final.
- Guardar commit de codigo exacto del run.
- Guardar comando exacto.
- Guardar version de datos exacta.

2. Prueba de “paridad de features”.
- Comparar estadisticas de entrada en train vs webcam (media, std, rango).
- Verificar que ambos usan exactamente el mismo preprocesado.

3. Prueba de “overfit controlado”.
- Entrenar con 32-64 muestras.
- Debe sobreajustar claramente.
- Si no sobreajusta, hay bug de pipeline.

## Bloque B (muy recomendado)

1. Crear script de evaluacion “single-letter webcam benchmark”.
- Sesiones cortas por letra (`a,b,c,...`).
- Matriz de confusion por letra.
- Curva de precision por umbral.

2. Ajustar decoder online.
- Consolidar captura guiada (`ESPACIO`) como modo principal de demo.
- Separar modo “continuo” y modo “guiado”.

3. Separar claramente dos objetivos:
- Modelo secuencial de palabras.
- Modelo clasificador de letra estatica (si se quiere demo de letra instantanea fiable).

## Bloque C (MLOps)

1. Checklist de release de modelo.
- arquitectura
- input_dim
- preprocesado
- metricas de validacion
- commit de entrenamiento

2. Validacion CI minima.
- Test de carga de checkpoint.
- Test de shape de entrada.
- Test de inferencia de humo (smoke test).

3. Registro de experimentos.
- una tabla por run con:
  - commit
  - dataset version
  - hiperparametros
  - CER/WER final
  - artefacto best

## 8) Estado actual despues de esta revision

- Se subio al repo la arquitectura `TCN_BiLSTM` explicita en `src/models/tcn_bilstm.py`.
- `model_loader.py` actua como factory.
- Se dejo demo webcam con modo guiado por `ESPACIO`.
- Se detecto como riesgo principal la desalineacion entre codigo actual y codigo real usado en el run final.

## 9) Recomendacion de trabajo para manana

1. Recuperar y versionar el `train.py` exacto que genero `archcmp2_tcn_bilstm_full_20260303`.
2. Repetir un run corto de control con ese codigo versionado.
3. Medir en benchmark guiado por letras antes de evaluar demo continua.

