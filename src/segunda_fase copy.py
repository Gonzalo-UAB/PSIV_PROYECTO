import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import time

from PIL import Image
from pathlib import Path


SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 8

DIR_TRAIN = Path(r"PSIV_PROYECTO\data\dataset_ASL_2\Train")
DIR_TEST = Path(r"PSIV_PROYECTO\data\dataset_ASL_2\Test")

EXTENSIONES = {".jpg", ".jpeg"}

letras = ["a","b","c","d","e","f","g","h","i","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y"]

dicResultados = {}
for i, let in enumerate(letras):
    dicResultados[let.upper()] = i


datagen = ImageDataGenerator(
    rotation_range=15,
    zoom_range=0.15,
    width_shift_range=0.15,
    height_shift_range=0.15
)


def cargar_imagen(ruta_imagen, size):
    img = Image.open(ruta_imagen)
    img = img.convert("L")
    img = img.resize(size)

    img_array = np.array(img, dtype=np.float32)
    img_array = img_array / 255.0

    return img_array


ini = time.perf_counter()

train_labels = np.zeros(88008, dtype=np.int64)
train_images = np.zeros((88008, SIZE[0], SIZE[1]), dtype=np.float32)

test_labels = np.zeros(720, dtype=np.int64)
test_images = np.zeros((720, SIZE[0], SIZE[1]), dtype=np.float32)


i = 0
for carpeta_letra in DIR_TRAIN.iterdir():
    if not carpeta_letra.is_dir():
        continue

    label = carpeta_letra.name
    print(f"Cargando letra train: {label}")

    for ruta_imagen in carpeta_letra.iterdir():
        if ruta_imagen.suffix.lower() not in EXTENSIONES:
            continue

        try:
            pixeles = cargar_imagen(ruta_imagen, SIZE)
            train_images[i] = pixeles
            train_labels[i] = dicResultados[label]
            i += 1
        except Exception as e:
            print(f"Error al cargar imagen train {i}: {e}")

num_train = i
train_images = train_images[:num_train]
train_labels = train_labels[:num_train]


i = 0
for carpeta_letra in DIR_TEST.iterdir():
    if not carpeta_letra.is_dir():
        continue

    label = carpeta_letra.name
    print(f"Cargando letra test: {label}")

    for ruta_imagen in carpeta_letra.iterdir():
        if ruta_imagen.suffix.lower() not in EXTENSIONES:
            continue

        try:
            pixeles = cargar_imagen(ruta_imagen, SIZE)
            test_images[i] = pixeles
            test_labels[i] = dicResultados[label]
            i += 1
        except Exception as e:
            print(f"Error al cargar imagen test {i}: {e}")

num_test = i
test_images = test_images[:num_test]
test_labels = test_labels[:num_test]


train_images = train_images[..., tf.newaxis]
test_images = test_images[..., tf.newaxis]


print("Train:", train_images.shape, train_labels.shape)
print("Test:", test_images.shape, test_labels.shape)

print("Labels train min/max:", train_labels.min(), train_labels.max())
print("Labels test min/max:", test_labels.min(), test_labels.max())


print("Construyendo el modelo...")

model = models.Sequential([
    layers.Input(shape=(64, 64, 1)),

    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.GlobalAveragePooling2D(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.4),

    layers.Dense(24, activation="softmax")
])


print("Compilando el modelo...")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6
    )
]


print("Iniciando el entrenamiento...")

history = model.fit(
    datagen.flow(train_images, train_labels, batch_size=BATCH_SIZE),
    epochs=EPOCHS,
    callbacks=callbacks
)


print("Evaluando en test...")

test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Loss final en test:", test_loss)
print("Accuracy final en test:", test_acc)


# Path("models").mkdir(exist_ok=True)
model.save("version5_3.keras")

final = time.perf_counter()
print(f"Tiempo total: {final - ini:.2f} segundos")

print("¡Entrenamiento completado!")