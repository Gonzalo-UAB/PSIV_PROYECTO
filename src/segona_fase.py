#DETECCIÓN DE GESTOS

#IMPORTS
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
IMG_PATH_BACKGROUND = str(BASE_DIR / 'data' / 'images' / 'background.png')
IMG_PATH_THUMB = str(BASE_DIR / 'data' / 'images' / 'thumbs_up.png')

img_bg = cv2.imread(IMG_PATH_BACKGROUND)
img_thumb = cv2.imread(IMG_PATH_THUMB)

if img_bg is None or img_thumb is None:
    raise ValueError("No se pudo cargar alguna imagen")

#DETECCIÓN DE GESTOS

#Preprocesamiento
#Pasar a YCrCb
img_ycrcb = cv2.cvtColor(img_thumb, cv2.COLOR_BGR2YCrCb)
img_bg_ycrcb = cv2.cvtColor(img_bg, cv2.COLOR_BGR2YCrCb)

#Hand like binary
img_bi = np.zeros_like(img_ycrcb[:,:,0], dtype=np.uint8)

for i in range(img_ycrcb.shape[0]):
    for j in range(img_ycrcb.shape[1]):
        Y, Cr, Cb = img_ycrcb[i, j]
        if 77 <= Cb <= 127 and 133 <= Cr <= 173:
            img_bi[i][j] = 1
        else:
            img_bi[i][j] = 0

        
img_bg_bi = np.zeros_like(img_bg_ycrcb[:,:,0], dtype=np.uint8)

for i in range(img_bg_ycrcb.shape[0]):
    for j in range(img_bg_ycrcb.shape[1]):
        Y, Cr, Cb = img_bg_ycrcb[i, j]
        if 77 <= Cb <= 127 and 133 <= Cr <= 173:
            img_bg_bi[i][j] = 1
        else:
            img_bg_bi[i][j] = 0

#Exclusive or para eliminar objetos de piel estaticos del fondo 
img_remove_static = cv2.bitwise_xor(img_bi, img_bg_bi)

#Quitar regiones de las mano gesticulando en el fondo 
img_and_not = cv2.bitwise_and(img_remove_static, cv2.bitwise_not(img_bg_bi))

#Closing de un opening
kernel = np.ones((5,5), np.uint8)
img_opening = cv2.morphologyEx(img_and_not, cv2.MORPH_OPEN, kernel)
img_closing = cv2.morphologyEx(img_opening, cv2.MORPH_CLOSE, kernel)

#Para cada región encontrar el centroide y el área, y eliminar las regiones pequeñas
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_closing)
clean_img = np.zeros_like(img_closing)
min_area = 250
for i in range(1, num_labels):
    if stats[i, cv2.CC_STAT_AREA] > min_area:
        clean_img[labels == i] = 255

centroids = centroids[1:]  # Ignorar el fondo
stats = stats[1:]  # Ignorar el fondo

result = cv2.cvtColor(img_thumb, cv2.COLOR_BGR2RGB)
contours, _ = cv2.findContours(clean_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

#PLOT
plt.figure(figsize=(15, 10))

plt.subplot(3, 3, 1)
plt.imshow(cv2.cvtColor(img_thumb, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(3, 3, 2)
plt.imshow(cv2.cvtColor(img_bg, cv2.COLOR_BGR2RGB))
plt.title('Background Image')
plt.axis('off')

plt.subplot(3, 3, 3)
plt.imshow(img_bi, cmap='gray')
plt.title('Binary Image')
plt.axis('off')

plt.subplot(3, 3, 4)
plt.imshow(img_bg_bi, cmap='gray')
plt.title('Background Binary Image')
plt.axis('off')

plt.subplot(3, 3, 5)
plt.imshow(img_remove_static, cmap='gray')
plt.title('Removed Static Objects')
plt.axis('off')

plt.subplot(3, 3, 6)
plt.imshow(img_and_not, cmap='gray')
plt.title('Final Mask')
plt.axis('off')

plt.subplot(3, 3, 7)
plt.imshow(img_closing, cmap='gray')
plt.title('Closed Open Image')
plt.axis('off')

plt.subplot(3, 3, 8) #Centroides
plt.imshow(clean_img, cmap='gray') 
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50, marker='x')
plt.title('Centroides de las regiones de las manos')
plt.axis('off')

plt.subplot(3, 3, 9) #Segmentos de las manos
plt.imshow(result, cmap='gray')
plt.title('Segmentos de las manos')
plt.axis('off')
plt.tight_layout()
plt.show()
