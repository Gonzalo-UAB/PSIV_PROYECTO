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

train_images, test_images = train_images / 255.0, test_images / 255.0

train_images = train_images[..., tf.newaxis]
test_images = test_images[..., tf.newaxis]

print("Construyendo el modelo...")
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    
    layers.Conv2D(32, (3, 3), activation='relu'),

    layers.MaxPooling2D((2, 2)), 
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    
    layers.Dense(128, activation='relu'),

    layers.Flatten(),

    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3), 
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    
    layers.Dense(25, activation='softmax') 
])

print("Compilando el modelo...")
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

print("Iniciando el entrenamiento...")

model.fit(datagen.flow(train_images, train_labels, batch_size=16), epochs=10, validation_data=(test_images, test_labels))

model.save("models/version3_0.keras")


print("¡Entrenamiento completado!")