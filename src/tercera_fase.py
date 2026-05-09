#DEEP LEARNING PARA AMERICAN SIGN LANGUAGE
import tensorflow as tf
from tensorflow.python.keras import layers, models
import numpy as np
import time


datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,   # Rota la imagen un poco
    zoom_range=0.1,      # Zoom aleatorio
    width_shift_range=0.1, 
    height_shift_range=0.1
)
# 1. Cargar y preparar los datos de ejemplo
print("Cargando el dataset Fashion MNIST...")
# fashion_mnist = tf.keras.datasets.fashion_mnist
# (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

ini = time.perf_counter()
train_labels = np.zeros(27455, dtype=np.int64)
train_images = np.zeros((27455, 28, 28), dtype=np.float32)

with open(r"PSIV_PROYECTO\data\dataset_mnist_ASL\sign_mnist_train.csv") as nH:
    nH.readline()

    for i, line in enumerate(nH):
        linia = line.strip().split(",")

        train_labels[i] = int(linia[0])

        pixels = np.array(linia[1:], dtype=np.float32)
        train_images[i] = pixels.reshape(28, 28)

test_labels = np.zeros(7172, dtype=np.int64)
test_images = np.zeros((7172, 28, 28), dtype=np.float32)

with open(r"PSIV_PROYECTO\data\dataset_mnist_ASL\sign_mnist_test.csv") as nH:
    nH.readline()

    for i, line in enumerate(nH):
        linia = line.strip().split(",")

        test_labels[i] = int(linia[0])

        pixels = np.array(linia[1:], dtype=np.float32)
        test_images[i] = pixels.reshape(28, 28)
                
final = time.perf_counter()


print(f"Final {final-ini}secs")
print(train_images[689])
print(test_images[639])

# Normalizar los valores de los píxeles para que estén entre 0 y 1 (facilita el entrenamiento)
train_images, test_images = train_images / 255.0, test_images / 255.0

# Añadir una dimensión extra para indicar que es un solo canal de color (escala de grises)
train_images = train_images[..., tf.newaxis]
test_images = test_images[..., tf.newaxis]

# 2. Construir la arquitectura del modelo (CNN)
print("Construyendo el modelo...")
model = models.Sequential([
    # Especificar la forma de entrada: 28x28 píxeles y 1 canal de color
    layers.Input(shape=(28, 28, 1)),
    
    # Capa convolucional: busca características (bordes, curvas) en áreas de 3x3 píxeles
    layers.Conv2D(32, (3, 3), activation='relu'),
    # MaxPooling: reduce a la mitad el tamaño de la imagen quedándose con las características más fuertes
    layers.MaxPooling2D((2, 2)), 
    
    # Segunda capa convolucional para detectar características más complejas
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Aplanar el mapa de características 2D a un vector 1D
    
    # Capa densa "Fully Connected" para procesar todo lo aprendido
    layers.Dense(128, activation='relu'),

    layers.Flatten(),

    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3), # Bajamos un poco de 0.5 a 0.3 para no castigar tanto el aprendizaje
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    
    # Capa de salida: 10 neuronas porque hay 10 categorías de ropa, usa softmax para dar probabilidades
    layers.Dense(25, activation='softmax') 
])

# 3. Compilar el modelo
print("Compilando el modelo...")
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Mostrar un resumen de la arquitectura que acabamos de crear
model.summary()

# 4. Entrenar el modelo
print("Iniciando el entrenamiento...")
# Entrenamos durante 5 "epochs" (pasadas completas por todos los datos)
model.fit(datagen.flow(train_images, train_labels, batch_size=16), epochs=10, validation_data=(test_images, test_labels))

model.save("version3_0.keras")


print("¡Entrenamiento completado!")