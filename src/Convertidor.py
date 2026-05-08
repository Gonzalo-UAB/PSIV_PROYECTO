from PIL import Image
import numpy as np

def preparar_imagen(ruta_imagen):
    # 1. Abrir la imagen
    img = Image.open(ruta_imagen)
    
    # 2. Convertir a escala de grises ('L')
    # Esto es vital porque el modelo se entrenó con 1 solo canal de color
    img = img.convert('L')
    
    # # 3. Redimensionar a 28x28 píxeles
    # # Usamos Resampling.LANCZOS para mantener la mejor calidad posible al encogerla
    # img = img.resize((28, 28), Image.Resampling.LANCZOS)
    
    # 4. Convertir a array de numpy
    img_array = np.array(img)
    
    # 5. Normalizar (0-255 -> 0.0-1.0)
    # img_array = img_array.astype('float32') / 255.0
    
    # 6. Invertir colores (Opcional)
    # MNIST usa fondo negro (0) y dibujo blanco (1). 
    # Si tu imagen es dibujo negro sobre fondo blanco, descomenta la siguiente línea:
    # img_array = 1.0 - img_array
    
    # 7. Expandir dimensiones para que sea (1, 28, 28, 1)    
    return img_array

# Uso:
# mi_imagen = preparar_imagen('mi_zapato.jpg')
# prediccion = model.predict(mi_imagen)