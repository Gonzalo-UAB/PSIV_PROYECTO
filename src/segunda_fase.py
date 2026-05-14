#DEEP LEARNING PARA AMERICAN SIGN LANGUAGE
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import time

from PIL import Image
from pathlib import Path

datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)

SIZE = (64, 64)

letras = ["a","b", "c","d","e","f","g","h","i","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y"]
nums = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
dicResultados = {}
for num, let in zip(nums,letras):
    dicResultados[let.upper()] =  num
DIR_TRAIN = Path(r"PSIV_PROYECTO\data\dataset_ASL_2\Train")
DIR_TEST = Path(r"PSIV_PROYECTO\data\dataset_ASL_2\Test")
      
EXTENSIONES = {".jpg", ".jpeg"} 


# datagen = preprocessing.image.ImageDataGenerator(
#     rotation_range=10,   # Rota la imagen un poco
#     zoom_range=0.1,      # Zoom aleatorio
#     width_shift_range=0.1, 
#     height_shift_range=0.1
# )

ini = time.perf_counter()
train_labels = np.zeros(88008, dtype=np.int64)
train_images = np.zeros((88008, 64,64), dtype=np.float32)

test_labels = np.zeros(720, dtype=np.int64)
test_images = np.zeros((720, 64,64), dtype=np.float32)

def cargar_imagen(ruta_imagen, size):

    img = Image.open(ruta_imagen)

    # Escala de grises
    img = img.convert("L")

    # Redimensionar por seguridad
    img = img.resize(size)

    # A numpy
    img_array = np.array(img, dtype=np.uint8)

    # 200x200 -> 40000
    img_array = img_array.flatten()

    return img_array

i = 0
for carpeta_letra in Path(DIR_TRAIN).iterdir():
        if not carpeta_letra.is_dir():
            continue

        label = carpeta_letra.name
        print(f"Cargando letra: {label}")

        # Recorrer imágenes
        for ruta_imagen in carpeta_letra.iterdir():
            if ruta_imagen.suffix.lower() not in EXTENSIONES:
                continue
            try:
            # Cargamos la imagen como array de numpy
                pixeles = cargar_imagen(ruta_imagen, SIZE) 
                
                pixeles = pixeles / 255.0

                # Aseguramos que sea un vector de tipo uint8 (0-255) para ahorrar espacio
                train_images[i]  = pixeles.reshape(SIZE)
                
                # Guardamos la etiqueta (como string o podrías convertirla a int)
                train_labels[i] = dicResultados[label]
                i+=1
            except:
                print(f"Error al carregar image {i}")
i = 0
for carpeta_letra in Path(DIR_TEST).iterdir():
        if not carpeta_letra.is_dir():
            continue

        label = carpeta_letra.name
        print(f"Cargando letra: {label}")

        # Recorrer imágenes
        for ruta_imagen in carpeta_letra.iterdir():
            if ruta_imagen.suffix.lower() not in EXTENSIONES:
                continue
            try:
            # Cargamos la imagen como array de numpy
                pixeles = cargar_imagen(ruta_imagen, SIZE) 
                
                pixeles = pixeles / 255.0

                # Aseguramos que sea un vector de tipo uint8 (0-255) para ahorrar espacio
                test_images[i]  = pixeles.reshape(SIZE)
                
                # Guardamos la etiqueta (como string o podrías convertirla a int)
                test_labels[i] = dicResultados[label]
                i+=1
            except:
                print(f"Error al carregar image {i}")

# train_images, test_images = train_images / 255.0, test_images / 255.0

train_images = train_images[..., tf.newaxis]
test_images = test_images[..., tf.newaxis]

print("Construyendo el modelo...")
model = models.Sequential([
    layers.Input(shape=(64, 64, 1)),
    
    layers.Conv2D(32, (3, 3), activation='relu'),

    layers.MaxPooling2D((2, 2)), 
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    
    # layers.Dense(128, activation='relu'),

    layers.Flatten(),

    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3), 
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    
    layers.Dense(24, activation='softmax') 
])

print("Compilando el modelo...")
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

print("Iniciando el entrenamiento...")

model.fit(datagen.flow(train_images, train_labels, batch_size=16), epochs=10, validation_data=(test_images, test_labels))

model.save("version4_0.keras")


print("¡Entrenamiento completado!")







# with h5py.File('tu_archivo.h5', 'r') as f:
#     X = f['X_train'][:] # El [:] carga los datos en RAM
#     y = f['Y_train'][:]
    


# with open(r"PSIV_PROYECTO\data\dataset_mnist_ASL\sign_mnist_train.csv") as nH:
#     nH.readline()

#     for i, line in enumerate(nH):
#         linia = line.strip().split(",")

#         train_labels[i] = int(linia[0])

#         pixels = np.array(linia[1:], dtype=np.float32)
#         train_images[i] = pixels.reshape(28, 28)



# with open(r"PSIV_PROYECTO\data\dataset_mnist_ASL\sign_mnist_test.csv") as nH:
#     nH.readline()

#     for i, line in enumerate(nH):
#         linia = line.strip().split(",")

#         test_labels[i] = int(linia[0])

#         pixels = np.array(linia[1:], dtype=np.float32)
#         test_images[i] = pixels.reshape(28, 28)
                
# final = time.perf_counter()


# print(f"Final {final-ini}secs")
# print(train_images[689])
# print(test_images[639])