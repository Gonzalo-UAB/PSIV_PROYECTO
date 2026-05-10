import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
IMG_PATH_BACKGROUND = str(BASE_DIR / 'data' / 'images' / 'background.png')
IMG_PATH_THUMB = str(BASE_DIR / 'data' / 'images' / 'thumbs_up.png')

img_bg = cv2.imread(IMG_PATH_BACKGROUND)
img_thumb = cv2.imread(IMG_PATH_THUMB)

if img_bg is None or img_thumb is None:
    raise ValueError("No se pudo cargar alguna imagen")

# =========================
# 1. Detección de cambio
# =========================
diff = cv2.absdiff(img_bg, img_thumb)
diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

_, mask_change = cv2.threshold(diff_gray, 18, 255, cv2.THRESH_BINARY)

kernel = np.ones((5, 5), np.uint8)
mask_change = cv2.morphologyEx(mask_change, cv2.MORPH_CLOSE, kernel)
mask_change = cv2.dilate(mask_change, kernel, iterations=2)

# =========================
# 2. Detección de piel (YCrCb)
# =========================
img_ycrcb = cv2.cvtColor(img_thumb, cv2.COLOR_BGR2YCrCb)

mask_skin = cv2.inRange(
    img_ycrcb,
    np.array([0, 133, 77], dtype=np.uint8),
    np.array([255, 173, 127], dtype=np.uint8)
)

# =========================
# 3. Mano = cambio + piel
# =========================
mask_hand = cv2.bitwise_and(mask_change, mask_skin)

# Limpieza
kernel = np.ones((7, 7), np.uint8)
mask_hand = cv2.morphologyEx(mask_hand, cv2.MORPH_OPEN, kernel)
mask_hand = cv2.morphologyEx(mask_hand, cv2.MORPH_CLOSE, kernel)

# =========================
# 4. Filtrar regiones pequeñas
# =========================
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_hand)

clean_mask = np.zeros_like(mask_hand)
min_area = 700

for i in range(1, num_labels):
    if stats[i, cv2.CC_STAT_AREA] > min_area:
        clean_mask[labels == i] = 255

# =========================
# 5. Dibujar resultado
# =========================
result = img_thumb.copy()
contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

# =========================
# 6. Plot final simple
# =========================
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(clean_mask, cmap='gray')
plt.title('Máscara mano')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title('Resultado')
plt.axis('off')

plt.tight_layout()
plt.show()
