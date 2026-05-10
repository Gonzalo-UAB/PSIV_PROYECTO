from PIL import Image
import numpy as np

def preparar_imagen(ruta_imagen):
    img = Image.open(ruta_imagen)
    
    # Esto es vital porque el modelo se entrenó con 1 solo canal de color
    img = img.convert('L')
    
    # # Usamos Resampling.LANCZOS para mantener la mejor calidad posible al encogerla
    # img = img.resize((28, 28), Image.Resampling.LANCZOS)
    
    img_array = np.array(img)

    img_array = img_array.astype('float32') / 255.0
    
    
    return img_array

