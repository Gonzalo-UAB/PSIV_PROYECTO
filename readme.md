# PSIV_PROYECTO

Proyecto de la asignatura de **Signal, Image and Video Processing** centrado en
reconocimiento de manos y gestos, evolucionando desde tecnicas clasicas de visión
por computador hasta modelos de deep learning con datos de imagen y datos esqueletales.

## Objetivo

El repositorio implementa 4 etapas de trabajo:

1. Deteccion de mano respecto a un fondo de referencia.
2. Segmentacion y analisis de gestos con tecnicas clasicas.
3. Clasificacion de letras ASL (MNIST ASL) con CNN.
4. Clasificacion de gestos/poses con features esqueletales y red densa.

## Requisitos

- Python 3.10+ (se recomienda usar el entorno `PSIV` ya configurado).
- Dependencias en `requirements.txt`.
- Datasets colocados en `data/` con la estructura mostrada abajo.

Instalacion:

```bash
pip install -r requirements.txt
```

Nota: en `requirements.txt` aparece `sickit-learn`; si falla esa linea, instalar:

```bash
pip install scikit-learn
```

## Estructura del proyecto

Arbol resumido:

```text
PSIV_PROYECTO/
├─ src/
│  ├─ primera_fase.py
│  ├─ segona_fase.py
│  ├─ tercera_fase.py
│  ├─ tercera_fase_pruebas.py
│  ├─ cuarta_fase.py
│  ├─ Convertidor.py
│  └─ Projecte.ipynb
├─ data/
│  ├─ dataset_mnist_ASL/
│  ├─ dataset_skeletal_hand_gesture/
│  ├─ images/
│  └─ videos/
├─ models/
│  ├─ version3_0.keras
│  ├─ gesture_model.keras
│  └─ scaler.pkl
├─ figs/
│  ├─ training_history.png
│  ├─ confusion_matrix.png
│  └─ classification_report.txt
├─ cache/
├─ miniProjecte-VC-PSIC.tex
└─ requirements.txt
```

## Que hace cada archivo de `src`

### `primera_fase.py`

Primera etapa (vision clasica):

- Carga imagen de fondo y una imagen con mano.
- Aplica diferencia absoluta (`absdiff`) para detectar cambio.
- Segmenta piel en espacio YCrCb.
- Combina ambas mascaras (cambio + piel).
- Limpia ruido con morfologia y connected components.
- Dibuja contornos y muestra resultados con `matplotlib`.

### `segona_fase.py`

Segunda etapa (gesto clasico):

- Convierte imagenes a RGB y YCrCb.
- Construye binarizacion de piel pixel a pixel.
- Elimina piel estatica del fondo con operaciones logicas.
- Limpia mascara con opening/closing.
- Detecta regiones de mano y centroides.
- Estima dedos levantados con convex hull/convexity defects.
- Extrae descriptores ORB y compara gestos por matches.
- Visualiza pipeline completo en una cuadricula de plots.

### `tercera_fase.py`

Tercera etapa (deep learning en ASL MNIST):

- Lee `sign_mnist_train.csv` y `sign_mnist_test.csv`.
- Normaliza datos y anade canal para CNN.
- Aplica `ImageDataGenerator` para augmentation.
- Entrena un modelo convolucional.
- Guarda el modelo en `models/version3_0.keras`.

### `tercera_fase_pruebas.py`

Script de inferencia para la tercera fase:

- Carga `models/version3_0.keras`.
- Usa `Convertidor.preparar_imagen` para preprocesar una imagen 28x28.
- Ejecuta prediccion y mapea el indice a letra ASL.
- Imprime la letra estimada por el modelo.

### `cuarta_fase.py`

Cuarta etapa (deep learning con datos esqueletales):

- Recorre `data/dataset_skeletal_hand_gesture/skeletal`.
- Parsea XML con `xmltodict` y extrae features de mano/dedos.
- Construye dataset en paralelo con `ProcessPoolExecutor`.
- Cachea dataset en `cache/skeletal_dataset.pkl`.
- Divide train/test, normaliza (`StandardScaler`) y entrena red densa.
- Evalua accuracy/loss.
- Guarda metricas y diagnosticos en `figs/`.
- Guarda modelo en `models/gesture_model.keras` y scaler en `models/scaler.pkl`.

### `Convertidor.py`

Utilidad de preprocesado para inferencia:

- Abre imagen con Pillow.
- Convierte a escala de grises.
- Normaliza a rango [0, 1].
- Devuelve array listo para usar en prediccion.

### `Projecte.ipynb`

Notebook de pruebas y experimentacion (prototipos de segmentacion y umbralizado).

## Flujo recomendado de ejecucion

Desde la raiz del repo:

```bash
python src/primera_fase.py
python src/segona_fase.py
python src/tercera_fase.py
python src/tercera_fase_pruebas.py
python src/cuarta_fase.py
```

## Datos y rutas esperadas

El codigo espera rutas de este estilo:

- `data/images/background.png`, `thumbs_up.png`, etc.
- `data/dataset_mnist_ASL/sign_mnist_train.csv` y `sign_mnist_test.csv`.
- `data/dataset_skeletal_hand_gesture/skeletal/<sujeto>/<subset>/<gesto>/*.xml`.

Si cambias nombres o rutas, ajusta las constantes `BASE_DIR`/`DATASET_PATH` en
cada script.

## Salidas generadas

- Modelos entrenados en `models/`.
- Graficas y reporte en `figs/`.
- Cache intermedio en `cache/`.

## Documento del proyecto

El informe tecnico se encuentra en `miniProjecte-VC-PSIC.tex` (con PDF generado en la raiz).