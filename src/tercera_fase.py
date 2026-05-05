#DEEP LEARNING PARA AMERICAN SIGN LANGUAGE
import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Cargar y preparar los datos de ejemplo
print("Cargando el dataset Fashion MNIST...")
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

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
    layers.Flatten(),
    
    # Capa densa "Fully Connected" para procesar todo lo aprendido
    layers.Dense(64, activation='relu'),
    
    # Capa de salida: 10 neuronas porque hay 10 categorías de ropa, usa softmax para dar probabilidades
    layers.Dense(10, activation='softmax') 
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
model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))

print("¡Entrenamiento completado!")