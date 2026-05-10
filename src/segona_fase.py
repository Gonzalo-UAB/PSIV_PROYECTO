#DETECCIÓN DE GESTOS

#IMPORTS
from pathlib import Path
from unittest import result
import matplotlib.pyplot as plt
import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
IMG_PATH_BACKGROUND = str(BASE_DIR / 'data' / 'images' / 'background.png')
#DETECCIÓN DE GESTOS

#Pasar a YCrCb
def to_rgb_and_ycrcb(img_path):
    if img_path is None:
        raise ValueError("No se pudo cargar la imagen")
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("No se pudo cargar la imagen")
    img_original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    return img_original, img_ycrcb

def represent_hand_in_binary(img_ycrcb):
    #Represent hand in binary with thresholds
    img_bi = np.zeros_like(img_ycrcb[:,:,0], dtype=np.uint8)

    for i in range(img_ycrcb.shape[0]):
        for j in range(img_ycrcb.shape[1]):
            Y, Cr, Cb = img_ycrcb[i, j]
            if 77 <= Cb <= 127 and 133 <= Cr <= 173:
                img_bi[i][j] = 1
            else:
                img_bi[i][j] = 0
    return img_bi

def preprocess_image(img_bi, img_bg_bi):
    #Exclusive or para eliminar objetos de piel estaticos del fondo 
    img_remove_static = cv2.bitwise_xor(img_bi, img_bg_bi)

    #Quitar regiones de las mano gesticulando en el fondo 
    img_and_not = cv2.bitwise_and(img_remove_static, cv2.bitwise_not(img_bg_bi))

    #Closing de un opening
    kernel_opening = np.ones((5,5), np.uint8)
    kernel_closing = np.ones((10,10), np.uint8)
    img_opening = cv2.morphologyEx(img_and_not, cv2.MORPH_OPEN, kernel_opening)
    img_closing = cv2.morphologyEx(img_opening, cv2.MORPH_CLOSE, kernel_closing)

    return img_remove_static, img_and_not,img_closing

def detect_gesture(img_closing, img_original):
    #Encontrar regiones de las manos con connected components
    #Para cada región encontrar el centroide y el área, y eliminar las regiones pequeñas
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_closing)
    clean_img = np.zeros_like(img_closing, dtype=np.uint8)
    min_area = 250
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > min_area:
            clean_img[labels == i] = 255

    centroids = centroids[1:]  # Ignorar el fondo
    stats = stats[1:]  # Ignorar el fondo

    result = img_original.copy()
    contours, _ = cv2.findContours(clean_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

    return result, centroids, clean_img, contours

def contar_dedos(contours):
    #Contar dedos levantados con Convex Hull
    num_fingers = 0
    ConvexHull = cv2.convexHull(contours[0], returnPoints=False)
    if ConvexHull is not None and len(ConvexHull) > 3:
        defects = cv2.convexityDefects(contours[0], ConvexHull)
        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(contours[0][s][0])
                end = tuple(contours[0][e][0])
                far = tuple(contours[0][f][0])
                if d > 1000:  # Umbral para contar un dedo levantado
                    num_fingers += 1
    return num_fingers

def orb_features(clean_img, result):
    #Detección de puntos clave con ORB
    orb = cv2.ORB_create(nfeatures=100)
    keypoints, descriptors = orb.detectAndCompute(clean_img, None)
    img_keypoints = cv2.drawKeypoints(result, keypoints, None, color=(0, 0, 255), flags=0)
    
    return img_keypoints, keypoints, descriptors

def Haar_features(clean_img, result):
    #Detección de puntos clave con Haar Cascades
    hand_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_hand.xml')
    hands = hand_cascade.detectMultiScale(clean_img, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in hands:
        cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    return result, hands

#Comparar con otro gesto

#PLOT
def plot_results(img1, img2, img3, img4, img5, img6, img7, img8, img9, centroids, num_fingers):
    plt.figure(figsize=(15, 10))

    plt.subplot(3, 3, 1)
    plt.imshow(img1)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(3, 3, 2)
    plt.imshow(img2)
    plt.title('Background Image')
    plt.axis('off')

    plt.subplot(3, 3, 3)
    plt.imshow(img3, cmap='gray')
    plt.title('Binary Image')
    plt.axis('off')

    plt.subplot(3, 3, 4)
    plt.imshow(img4, cmap='gray')
    plt.title('Background Binary Image')
    plt.axis('off')

    plt.subplot(3, 3, 5)
    plt.imshow(img5, cmap='gray')
    plt.title('Removed Static Objects')
    plt.axis('off')

    plt.subplot(3, 3, 6)
    plt.imshow(img6, cmap='gray')
    plt.title('Final Mask')
    plt.axis('off')

    plt.subplot(3, 3, 7)
    plt.imshow(img7, cmap='gray')
    plt.title('Closed Open Image')
    plt.axis('off')

    plt.subplot(3, 3, 8) #Centroides
    plt.imshow(img8, cmap='gray') 
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50, marker='x')
    plt.title('Centroides de las regiones de las manos')
    plt.axis('off')

    plt.subplot(3, 3, 9) #Segmentos de las manos
    plt.imshow(img9)
    plt.title(f'Segmentos de las manos con {num_fingers} dedos levantados')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    img_bg, img_bg_ycrcb = to_rgb_and_ycrcb(IMG_PATH_BACKGROUND)
    img_bg_bi = represent_hand_in_binary(img_bg_ycrcb)

    llista_imgs = ["thumbs_down", "thumbs_up", "palm", "ok"]
    llista_descriptors = []
    #Procesar cada imagen de la lista
    for img in llista_imgs:
        img_path = str(BASE_DIR / 'data' / 'images' / f'{img}.png')
        img_original, img_ycrcb = to_rgb_and_ycrcb(img_path)
        img_bi = represent_hand_in_binary(img_ycrcb)
        img_remove_static, img_and_not, img_closing = preprocess_image(img_bi, img_bg_bi)
        result, centroids, clean_img, contours = detect_gesture(img_closing, img_original)
        num_fingers = contar_dedos(contours)
        img_keypoints, keypoints, descriptors = orb_features(clean_img, result)
        llista_descriptors.append(descriptors)

        plot_results(img_original, img_bg, img_bi, img_bg_bi, img_remove_static, img_and_not, img_closing, result, img_keypoints, centroids, num_fingers)

    # Comparar los descriptores de las imágenes con ORB
    for i in range(len(llista_descriptors)):
        for j in range(i + 1, len(llista_descriptors)):
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(llista_descriptors[i], llista_descriptors[j])
            matches = sorted(matches, key=lambda x: x.distance)
            print(f'Matches entre {llista_imgs[i]} y {llista_imgs[j]}: {len(matches)}')
    
    print("Detección de manos y comparación de descriptores finalizada!")

if __name__ == "__main__":
    main()