import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
import numpy as np
import time
from Convertidor import preparar_imagen

model = load_model("Cuarta_version.keras")
test_labels = np.zeros(7172, dtype=np.int64)
test_images = np.zeros((7172, 28, 28), dtype=np.float32)

with open(r"PSIV_PROYECTO\data\dataset_mnist_ASL\sign_mnist_test.csv") as nH:
    nH.readline()

    for i, line in enumerate(nH):
        linia = line.strip().split(",")

        test_labels[i] = int(linia[0])

        pixels = np.array(linia[1:], dtype=np.float32)
        test_images[i] = pixels.reshape(28, 28)



# Tomamos una imagen del set de prueba como ejemplo
img = preparar_imagen('a_28x28.jpg')

# Los modelos de Keras están optimizados para hacer predicciones en "lotes" (batches)
# Por eso añadimos una dimensión extra para que sea (1, 28, 28, 1)
img_batch = (np.expand_dims(img, 0))

# Realizar la predicción
predicciones = model.predict(img_batch)

# La salida es un array con 10 probabilidades. Buscamos el índice con la mayor.
indice_clase = np.argmax(predicciones[0])

print(f"El modelo predice que es la clase: {indice_clase}")