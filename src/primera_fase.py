#DETECCIÓN DE MANOS
import matplotlib.pyplot as plt
import cv2

IMG_PATH_BACKGROUND = 'data/images/background.png'
IMG_PATH_THUMB = 'data/images/thumb.png'
img_bg = cv2.imread(IMG_PATH_BACKGROUND)
img_thumb = cv2.imread(IMG_PATH_THUMB)

img_bg_gray = cv2.cvtColor(img_bg, cv2.COLOR_BGR2GRAY)
img_thumb_gray = cv2.cvtColor(img_thumb, cv2.COLOR_BGR2GRAY)

img_bg_gray = cv2.GaussianBlur(img_bg_gray, (5, 5), 0)
img_thumb_gray = cv2.GaussianBlur(img_thumb_gray, (5, 5), 0)
img_substract = cv2.absdiff(img_bg_gray, img_thumb_gray)

#OTSU THRESHOLD
img_thresh = cv2.threshold(img_substract, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

plt.imshow(img_thresh[1], cmap='gray')
plt.title('Otsu Thresholding')
