import tensorflow as tf
from tensorflow.python.keras import layers, models
from tensorflow.python.keras.models import load_model
import numpy as np
import time
from Convertidor import preparar_imagen
letras = ["a","b", "c","d","e","f","g","h","i","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y"]
nums = [0,1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
dicResultados = {}
for num, let in zip(nums,letras):
    dicResultados[num] = let

model = load_model("models/version3_0.keras")
test_labels = np.zeros(7172, dtype=np.int64)
test_images = np.zeros((7172, 28, 28), dtype=np.float32)

# with open(r"PSIV_PROYECTO\data\dataset_mnist_ASL\sign_mnist_test.csv") as nH:
#     nH.readline()

#     for i, line in enumerate(nH):
#         linia = line.strip().split(",")

#         test_labels[i] = int(linia[0])

#         pixels = np.array(linia[1:], dtype=np.float32)
#         test_images[i] = pixels.reshape(28, 28)



img = preparar_imagen('r_blanco_28x28.png')

img_batch = (np.expand_dims(img, 0))

# Realizar la predicción
predicciones = model.predict(img_batch)

indice_clase = int(np.argmax(predicciones[0]))

print(f"El modelo predice que es la clase: {dicResultados[indice_clase]}")