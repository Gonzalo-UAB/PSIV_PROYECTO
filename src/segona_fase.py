#DETECCIÓN DE GESTOS

#IMPORTS
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
IMG_PATH_BACKGROUND = str(BASE_DIR / 'data' / 'images' / 'background.png')
IMG_PATH_THUMB = str(BASE_DIR / 'data' / 'images' / 'thumbs_up.png')
IMG_PATH_PALM = str(BASE_DIR / 'data' / 'images' / 'palm.png')
img_bg = cv2.imread(IMG_PATH_BACKGROUND)
img_thumb = cv2.imread(IMG_PATH_THUMB)
img_palm = cv2.imread(IMG_PATH_PALM)

if img_bg is None or img_thumb is None or img_palm is None:
    raise ValueError("No se pudo cargar alguna imagen")

#DETECCIÓN DE GESTOS

#Preprocesamiento
#Pasar a YCrCb
img_original = cv2.cvtColor(img_palm, cv2.COLOR_BGR2RGB)
img_ycrcb = cv2.cvtColor(img_palm, cv2.COLOR_BGR2YCrCb)
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
kernel_opening = np.ones((5,5), np.uint8)
kernel_closing = np.ones((15,15), np.uint8)
img_opening = cv2.morphologyEx(img_and_not, cv2.MORPH_OPEN, kernel_opening)
img_closing = cv2.morphologyEx(img_opening, cv2.MORPH_CLOSE, kernel_closing)

#Para cada región encontrar el centroide y el área, y eliminar las regiones pequeñas
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_closing)
clean_img = np.zeros_like(img_closing)
min_area = 250
for i in range(1, num_labels):
    if stats[i, cv2.CC_STAT_AREA] > min_area:
        clean_img[labels == i] = 255

centroids = centroids[1:]  # Ignorar el fondo
stats = stats[1:]  # Ignorar el fondo

result = img_original.copy()
contours, _ = cv2.findContours(clean_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

#Contar dedos levantados con Convex Hull
num_fingers = 0
ConvexHull = cv2.convexHull(contours[0], returnPoints=False)
if ConvexHull is not None and len(ConvexHull) > 3:
    defects = cv2.convexityDefects(contours[0], ConvexHull)
    if defects is not None:
        print(f"Defect depths: {defects[:, 0, 3]}")  # Print all defect depths
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            print(f"Defect {i}: depth={d}")  # See individual values
    if defects is not None:
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(contours[0][s][0])
            end = tuple(contours[0][e][0])
            far = tuple(contours[0][f][0])
            if d > 3000:  # Umbral para contar un dedo levantado
                num_fingers += 1


#PLOT
plt.figure(figsize=(15, 10))

plt.subplot(3, 3, 1)
plt.imshow(img_original)
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
plt.imshow(result)
plt.title(f'Segmentos de las manos con {num_fingers} dedos levantados')
plt.axis('off')
plt.tight_layout()
plt.show()
