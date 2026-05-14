from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py

# =========================
# CONFIG
# =========================

DIR_TRAIN = Path(r"PSIV_PROYECTO\data\dataset_ASL_2\Train")
DIR_TEST = Path(r"PSIV_PROYECTO\data\dataset_ASL_2\Test")
      
SIZE = (200, 200)

EXTENSIONES = {".jpg", ".jpeg"}   # tus imágenes


# =========================
# CARGAR UNA IMAGEN
# =========================

def cargar_imagen(ruta_imagen, size=SIZE):

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


# =========================
# DATASET -> CSV
# =========================

def imagenes_a_h5py(carpeta_dataset, salida_h5, size=SIZE):
    imagenes = []
    etiquetas = []

    # Recorrer carpetas (A, B, C...)
    for carpeta_letra in Path(carpeta_dataset).iterdir():
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
                pixeles = cargar_imagen(ruta_imagen, size) 
                
                # Aseguramos que sea un vector de tipo uint8 (0-255) para ahorrar espacio
                imagenes.append(pixeles.astype('uint8'))
                
                # Guardamos la etiqueta (como string o podrías convertirla a int)
                etiquetas.append(label.encode('utf-8')) 

            except Exception as e:
                print(f"Error con {ruta_imagen}: {e}")
        # break

    print("Carga de datos en memoria completada. Guardando en HDF5...")

    # Convertimos las listas a arrays de NumPy
    X = np.array(imagenes)
    y = np.array(etiquetas)

    # Crear el archivo H5
    with h5py.File(salida_h5, 'w') as f:
        # Guardamos las imágenes con compresión gzip para reducir el peso drásticamente
        f.create_dataset('X', data=X, compression="gzip", chunks=True)
        
        # Guardamos las etiquetas
        f.create_dataset('Y', data=y)

    print(f"¡Proceso terminado! Archivo guardado en: {salida_h5}")
    print(f"Total de imágenes procesadas: {len(X)}")


def guardar_datos(nombreArchivo, imagenes, etiquetas):
    with h5py.File('dataset_letras.h5', 'w') as f:
        f.create_dataset('X_train', data=imagenes, compression="gzip", chunks=True)
        f.create_dataset('Y_train', data=etiquetas)
# =========================
# GENERAR CSVs
# =========================

# df_train = imagenes_a_h5py(
#     DIR_TRAIN,
#     "train.h5"
# )

# df_test = imagenes_a_h5py(
#     DIR_TEST,
#     "test.h5"
# )

def prueba_lectura(dataset_path):

    print("\n=== PRUEBA DE LECTURA ===\n")

    for carpeta_letra in dataset_path.iterdir():

        if not carpeta_letra.is_dir():
            continue

        letra = carpeta_letra.name

        # Buscar primera imagen válida
        imagenes = [
            img for img in carpeta_letra.iterdir()
            if img.suffix.lower() in EXTENSIONES
        ]

        if not imagenes:
            print(f"No hay imágenes en {letra}")
            continue

        ruta_imagen = imagenes[0]

        try:

            img_array = cargar_imagen(ruta_imagen)

            print(f"Letra: {letra}")
            print(f"Archivo: {ruta_imagen.name}")
            print(f"Shape: {img_array.shape}")
            print(f"Tipo: {img_array.dtype}")
            print(f"Primeros píxeles: {img_array.flatten()[:10]}")
            print("-" * 40)

            # Mostrar imagen
            plt.imshow(img_array.reshape(SIZE), cmap="gray")
            plt.title(f"Letra: {letra}")
            plt.axis("off")
            plt.show()

        except Exception as e:

            print(f"Error en {ruta_imagen}: {e}")


# def main():

#     prueba_lectura(DIR_TRAIN)


if __name__ == "__main__":
    contador=0
    for carpeta_letra in Path(DIR_TEST).iterdir():
        if not carpeta_letra.is_dir():
            continue

        label = carpeta_letra.name
        print(f"Cargando letra: {label}")

        # Recorrer imágenes
        for ruta_imagen in carpeta_letra.iterdir():
            if ruta_imagen.suffix.lower()  in EXTENSIONES:
                contador += 1
    print(f"En el dir test hay {contador} imagenes")

