import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np
import pathlib
import pickle
import sys
import time
from urllib.request import urlretrieve
import json
from datetime import datetime
import csv

# Rutas
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / 'models' / 'gesture_model.keras'
SCALER_PATH = BASE_DIR / 'models' / 'scaler.pkl'
HAND_LANDMARKER_TASK_PATH = BASE_DIR / 'cache' / 'hand_landmarker.task'
HAND_LANDMARKER_TASK_URL = (
    'https://storage.googleapis.com/mediapipe-models/hand_landmarker/'
    'hand_landmarker/float16/1/hand_landmarker.task'
)

# Directorios de logging
LOGS_DIR = BASE_DIR / 'logs'
SKELETAL_DATA_DIR = LOGS_DIR / 'skeletal_data'
PREDICTIONS_LOG_DIR = LOGS_DIR / 'predictions'
FRAMES_DIR = LOGS_DIR / 'frames'

for dir_path in [LOGS_DIR, SKELETAL_DATA_DIR, PREDICTIONS_LOG_DIR, FRAMES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Timestamp para esta sesión
SESSION_TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
SKELETAL_LOG_FILE = SKELETAL_DATA_DIR / f'skeletal_{SESSION_TIMESTAMP}.csv'
PREDICTIONS_LOG_FILE = PREDICTIONS_LOG_DIR / f'predictions_{SESSION_TIMESTAMP}.csv'
FRAMES_SAVE_INTERVAL = 10  # Guardar frame cada N predicciones

# Mapeo de gestos a índices (debe coincidir con tercera_fase.py)
GESTURE_LABELS = {
    0: '02_l', 
    1: '04_fist_moved', 
    2: '06_index', 
    3: '07_ok', 
    4: '09_c',
    5: '11_heavy', 
    6: '12_hang', 
    7: '13_two', 
    8: '14_three', 
    9: '15_four',
    10: '16_five', 
    11: '19_down', 
    12: '20_left', 
    13: '21_right', 
    14: '22_up'
}

# Cargar modelo y scaler
print(f"Cargando modelo desde: {MODEL_PATH}")
model = tf.keras.models.load_model(str(MODEL_PATH))

print(f"Cargando scaler desde: {SCALER_PATH}")
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

EXPECTED_FEATURES = int(getattr(scaler, 'n_features_in_', 96))
print(f"Scaler espera {EXPECTED_FEATURES} features")

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


def ensure_hand_landmarker_model(task_path):
    task_path.parent.mkdir(parents=True, exist_ok=True)
    if task_path.exists():
        return

    print(f"Descargando modelo de HandLandmarker en: {task_path}")
    try:
        urlretrieve(HAND_LANDMARKER_TASK_URL, str(task_path))
    except Exception as e:
        raise RuntimeError(
            'No se pudo descargar hand_landmarker.task. '
            'Comprueba tu conexión o descarga manualmente el archivo en:\n'
            f'  {task_path}\n'
            f'URL: {HAND_LANDMARKER_TASK_URL}'
        ) from e


def initialize_hand_detector():
    """
    Inicializa el detector de manos:
    1) Usa la API clásica `solutions` si está disponible.
    2) Si no, usa la Tasks API (compatible con MediaPipe recientes).
    """
    if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'hands'):
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        return {
            'mode': 'solutions',
            'hands': hands,
            'mp_hands': mp_hands,
            'mp_drawing': mp.solutions.drawing_utils
        }

    ensure_hand_landmarker_model(HAND_LANDMARKER_TASK_PATH)

    try:
        from mediapipe.tasks import python as mp_python_tasks
        from mediapipe.tasks.python import vision
    except Exception as e:
        raise ImportError(
            'No se pudo importar la Tasks API de MediaPipe.\n'
            f'Versión detectada: {getattr(mp, "__version__", "desconocida")}\n'
            f'Python actual: {sys.executable}'
        ) from e

    options = vision.HandLandmarkerOptions(
        base_options=mp_python_tasks.BaseOptions(model_asset_path=str(HAND_LANDMARKER_TASK_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    hand_landmarker = vision.HandLandmarker.create_from_options(options)

    return {
        'mode': 'tasks',
        'landmarker': hand_landmarker
    }


hand_detector = initialize_hand_detector()

def _compute_grab_and_pinch(landmarks):
    """Aproxima grab/pinch strength a partir de distancias entre landmarks."""
    wrist = landmarks[0]
    middle_mcp = landmarks[9]
    palm_scale = np.linalg.norm([
        middle_mcp.x - wrist.x,
        middle_mcp.y - wrist.y,
        middle_mcp.z - wrist.z
    ]) + 1e-6

    finger_pairs = [(4, 2), (8, 5), (12, 9), (16, 13), (20, 17)]
    tip_to_mcp = []
    for tip_idx, mcp_idx in finger_pairs:
        tip = landmarks[tip_idx]
        mcp = landmarks[mcp_idx]
        dist = np.linalg.norm([tip.x - mcp.x, tip.y - mcp.y, tip.z - mcp.z])
        tip_to_mcp.append(dist / palm_scale)

    # Mano cerrada -> distancias pequeñas -> grab cercano a 1
    grab_strength = float(np.clip(1.0 - np.mean(tip_to_mcp) / 1.6, 0.0, 1.0))

    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    pinch_dist = np.linalg.norm([
        thumb_tip.x - index_tip.x,
        thumb_tip.y - index_tip.y,
        thumb_tip.z - index_tip.z
    ]) / palm_scale
    pinch_strength = float(np.clip(1.0 - pinch_dist / 1.2, 0.0, 1.0))

    return grab_strength, pinch_strength, palm_scale


def _build_image_features_from_mediapipe(landmarks, confidence):
    """
    Construye 44 features compatibles con tercera_fase.py para una imagen.
    MediaPipe proporciona valores normalizados (0-1), necesitamos denormalizarlos
    para coincidir con la escala de entrenamiento (valores en píxeles ~500).
    """
    # Factor de escala para desnormalizar (MediaPipe: 0-1 -> coordenadas de imagen: 0-~500)
    SCALE_FACTOR = 500.0  # Aproximación basada en resolución típica de imágenes de entrenamiento
    
    # Calcular centroide real de la mano (promedio de todos los landmarks)
    center_x = np.mean([lm.x for lm in landmarks]) * SCALE_FACTOR
    center_y = np.mean([lm.y for lm in landmarks]) * SCALE_FACTOR
    center_z = np.mean([lm.z for lm in landmarks]) * SCALE_FACTOR
    
    # Wrist (landmark 0)
    wrist_x = landmarks[0].x * SCALE_FACTOR
    wrist_y = landmarks[0].y * SCALE_FACTOR
    wrist_z = landmarks[0].z * SCALE_FACTOR

    skeleton_points = [center_x, center_y, center_z, wrist_x, wrist_y, wrist_z]

    # Orden de dedos (Thumb, Index, Middle, Ring, Pinky): (tip_idx, mcp_idx)
    # MediaPipe: Thumb(4,3), Index(8,7), Middle(12,11), Ring(16,15), Pinky(20,19)
    tip_mcp_indices = [(4, 3), (8, 7), (12, 11), (16, 15), (20, 19)]
    extended = []
    grab_strength, pinch_strength, palm_scale = _compute_grab_and_pinch(landmarks)

    for tip_idx, mcp_idx in tip_mcp_indices:
        tip = landmarks[tip_idx]
        mcp = landmarks[mcp_idx]
        tip_x = tip.x * SCALE_FACTOR
        tip_y = tip.y * SCALE_FACTOR
        tip_z = tip.z * SCALE_FACTOR
        mcp_x = mcp.x * SCALE_FACTOR
        mcp_y = mcp.y * SCALE_FACTOR
        mcp_z = mcp.z * SCALE_FACTOR
        
        skeleton_points.extend([tip_x, tip_y, tip_z, mcp_x, mcp_y, mcp_z])

        # Heurística de dedo extendido basada en distancia tip-mcp normalizada
        dist = np.linalg.norm([tip.x - mcp.x, tip.y - mcp.y, tip.z - mcp.z]) / palm_scale
        extended.append(1.0 if dist > 0.6 else 0.0)

    image_features = skeleton_points + [float(confidence), grab_strength, pinch_strength] + extended
    return image_features


def extract_features_from_mediapipe(landmarks, hand_side='Unknown', handedness_score=0.0):
    """
    Extrae 44 features de una sola mano desde los landmarks de MediaPipe.
    """
    return _build_image_features_from_mediapipe(landmarks, handedness_score)


def build_features_for_both_hands(right_landmarks=None, right_score=0.0, left_landmarks=None, left_score=0.0):
    """
    Construye features combinadas para ambas manos (RightImage + LeftImage).
    Sigue el orden: RightImage (44 features) + LeftImage (44 features) = 88 features
    """
    # Extraer features de mano derecha o llenar con ceros
    if right_landmarks is not None:
        right_features = extract_features_from_mediapipe(right_landmarks, 'Right', right_score)
    else:
        right_features = [0.0] * 44
    
    # Extraer features de mano izquierda o llenar con ceros
    if left_landmarks is not None:
        left_features = extract_features_from_mediapipe(left_landmarks, 'Left', left_score)
    else:
        left_features = [0.0] * 44
    
    # Combinar: RightImage + LeftImage
    full_features = right_features + left_features
    
    if len(full_features) < EXPECTED_FEATURES:
        full_features.extend([0.0] * (EXPECTED_FEATURES - len(full_features)))
    elif len(full_features) > EXPECTED_FEATURES:
        full_features = full_features[:EXPECTED_FEATURES]
    
    features = np.array(full_features, dtype=np.float32)
    return features.reshape(1, -1)


def predict_gesture(features):
    """
    Hace predicción con el modelo
    """
    # Normalizar features usando el scaler
    features_scaled = scaler.transform(features)
    
    # Hacer predicción
    prediction = model.predict(features_scaled, verbose=0)
    
    # Obtener clase con mayor probabilidad
    class_idx = np.argmax(prediction[0])
    confidence = prediction[0][class_idx]
    
    gesture_name = GESTURE_LABELS.get(class_idx, 'Desconocido')
    
    return gesture_name, confidence, class_idx


def draw_hand_skeleton(image, landmarks, w, h, mirrored=False):
    """Dibuja landmarks y conexiones para la salida de MediaPipe."""
    points = []
    for lm in landmarks:
        x = int(lm.x * w)
        if mirrored:
            x = w - x
        y = int(lm.y * h)
        points.append((x, y))
        cv2.circle(image, (x, y), 3, (255, 200, 0), -1)

    for start, end in HAND_CONNECTIONS:
        x1, y1 = points[start]
        x2, y2 = points[end]
        cv2.line(image, (x1, y1), (x2, y2), (0, 180, 255), 2)


def get_handedness_label(landmarker_result, index):
    if not landmarker_result.handedness or index >= len(landmarker_result.handedness):
        return 'Unknown'

    categories = landmarker_result.handedness[index]
    if not categories:
        return 'Unknown'

    category = categories[0]
    return getattr(category, 'category_name', 'Unknown')


def save_skeletal_data(timestamp, right_landmarks=None, right_score=0.0, left_landmarks=None, left_score=0.0):
    """Guarda los datos esqueletales de ambas manos en CSV."""
    try:
        # Preparar datos
        row = {
            'timestamp': timestamp,
            'right_hand_detected': right_landmarks is not None,
            'right_confidence': right_score if right_landmarks else 0.0,
            'left_hand_detected': left_landmarks is not None,
            'left_confidence': left_score if left_landmarks else 0.0
        }
        
        # Agregar landmarks de mano derecha
        if right_landmarks:
            for idx, lm in enumerate(right_landmarks):
                row[f'right_lm_{idx}_x'] = lm.x
                row[f'right_lm_{idx}_y'] = lm.y
                row[f'right_lm_{idx}_z'] = lm.z
        
        # Agregar landmarks de mano izquierda
        if left_landmarks:
            for idx, lm in enumerate(left_landmarks):
                row[f'left_lm_{idx}_x'] = lm.x
                row[f'left_lm_{idx}_y'] = lm.y
                row[f'left_lm_{idx}_z'] = lm.z
        
        # Escribir o agregar a CSV
        file_exists = SKELETAL_LOG_FILE.exists()
        with open(SKELETAL_LOG_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
    except Exception as e:
        print(f"Error guardando datos esqueletales: {e}")


def log_prediction(timestamp, gesture_name, confidence, gesture_idx, right_detected, left_detected):
    """Registra la predicción del modelo."""
    try:
        row = {
            'timestamp': timestamp,
            'gesture': gesture_name,
            'confidence': confidence,
            'class_idx': gesture_idx,
            'right_hand': right_detected,
            'left_hand': left_detected
        }
        
        # Escribir o agregar a CSV
        file_exists = PREDICTIONS_LOG_FILE.exists()
        with open(PREDICTIONS_LOG_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
    except Exception as e:
        print(f"Error registrando predicción: {e}")


def save_frame_with_skeleton(image, timestamp, frame_number):
    """Guarda un frame con los skeletons dibujados."""
    try:
        frame_filename = FRAMES_DIR / f'frame_{SESSION_TIMESTAMP}_{frame_number:06d}_{timestamp.replace(":", "").replace(".", "")}.jpg'
        cv2.imwrite(str(frame_filename), image)
    except Exception as e:
        print(f"Error guardando frame: {e}")


# Inicializar captura de video
cap = cv2.VideoCapture(0)

# Obtener propiedades del video
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Video: {width}x{height} @ {fps} FPS")
print("Presiona ESC para salir")
print(f"\nLogs guardándose en: {LOGS_DIR}")
print(f"  - Datos esqueletales: {SKELETAL_LOG_FILE}")
print(f"  - Predicciones: {PREDICTIONS_LOG_FILE}")
print(f"  - Frames: {FRAMES_DIR}\n")

frame_count = 0

try:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Error leyendo video")
            break

        frame_count += 1
        current_timestamp = datetime.now().isoformat()

        # Voltear horizontalmente para selfie view
        h, w, c = image.shape
        display_image = cv2.flip(image, 1)

        # MediaPipe requiere RGB, OpenCV usa BGR
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if hand_detector['mode'] == 'solutions':
            results = hand_detector['hands'].process(image_rgb)
            if results.multi_hand_landmarks and results.multi_handedness:
                # Organizar manos por tipo (Left/Right)
                hands_by_side = {}
                
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    landmarks = hand_landmarks.landmark
                    hand_side = handedness.classification[0].label
                    hand_score = float(getattr(handedness.classification[0], 'score', 0.0))
                    
                    # Dibujar skeleton
                    draw_hand_skeleton(display_image, landmarks, w, h, mirrored=True)
                    
                    hands_by_side[hand_side] = {
                        'landmarks': landmarks,
                        'score': hand_score
                    }
                
                # Extraer features de ambas manos
                right_landmarks = hands_by_side.get('Right', {}).get('landmarks')
                right_score = hands_by_side.get('Right', {}).get('score', 0.0)
                left_landmarks = hands_by_side.get('Left', {}).get('landmarks')
                left_score = hands_by_side.get('Left', {}).get('score', 0.0)
                
                features = build_features_for_both_hands(right_landmarks, right_score, left_landmarks, left_score)
                
                # Hacer una predicción con ambas manos
                gesture_name, confidence, class_idx = predict_gesture(features)
                
                # Guardar datos esqueletales y predicción
                save_skeletal_data(current_timestamp, 
                                 right_landmarks=hands_by_side.get('Right', {}).get('landmarks'),
                                 right_score=hands_by_side.get('Right', {}).get('score', 0.0),
                                 left_landmarks=hands_by_side.get('Left', {}).get('landmarks'),
                                 left_score=hands_by_side.get('Left', {}).get('score', 0.0))
                
                log_prediction(current_timestamp, gesture_name, float(confidence), int(class_idx),
                             'Right' in hands_by_side, 'Left' in hands_by_side)
                
                # Imprimir en consola cada 30 frames
                if frame_count % 30 == 0:
                    print(f"[Frame {frame_count}] {gesture_name}: {confidence:.3f}")
                
                # Guardar frame cada FRAMES_SAVE_INTERVAL frames
                if frame_count % FRAMES_SAVE_INTERVAL == 0:
                    save_frame_with_skeleton(display_image, current_timestamp.split('T')[1], frame_count)
                
                # Dibujar resultados en todas las manos detectadas
                for hand_side, hand_data in hands_by_side.items():
                    landmarks = hand_data['landmarks']
                    
                    # Obtener posición del bounding box para mostrar el texto
                    x_coords = [lm.x * w for lm in landmarks]
                    y_coords = [lm.y * h for lm in landmarks]

                    x_min, x_max = int(min(x_coords)), int(max(x_coords))
                    y_min, y_max = int(min(y_coords)), int(max(y_coords))

                    # Agregar margen
                    margin = 10
                    x_min = max(0, x_min - margin)
                    y_min = max(0, y_min - margin)
                    x_max = min(w, x_max + margin)
                    y_max = min(h, y_max + margin)

                    # Espejar coordenadas X para dibujar sobre la imagen volteada
                    x_min_m = w - x_max
                    x_max_m = w - x_min

                    # Dibujar rectángulo alrededor de la mano
                    cv2.rectangle(display_image, (x_min_m, y_min), (x_max_m, y_max), (0, 255, 0), 2)

                    # Mostrar gesto y confianza
                    text_gesture = f"{gesture_name}"
                    text_confidence = f"Conf: {confidence:.2f}"

                    cv2.putText(display_image, f"{hand_side}: {text_gesture}",
                               (x_min_m, y_min - 30), cv2.FONT_HERSHEY_SIMPLEX,
                               0.7, (0, 255, 0), 2)
                    cv2.putText(display_image, text_confidence,
                               (x_min_m, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX,
                               0.6, (0, 255, 0), 1)
        else:
            timestamp_ms = int(time.time() * 1000)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            results = hand_detector['landmarker'].detect_for_video(mp_image, timestamp_ms)

            if results.hand_landmarks:
                # Organizar manos por tipo (Left/Right)
                hands_by_side = {}
                
                for idx, landmarks in enumerate(results.hand_landmarks):
                    draw_hand_skeleton(display_image, landmarks, w, h, mirrored=True)
                    hand_side = get_handedness_label(results, idx)
                    hand_score = 0.0
                    if results.handedness and idx < len(results.handedness) and results.handedness[idx]:
                        hand_score = float(getattr(results.handedness[idx][0], 'score', 0.0))
                    
                    hands_by_side[hand_side] = {
                        'landmarks': landmarks,
                        'score': hand_score
                    }
                
                # Extraer features de ambas manos
                right_landmarks = hands_by_side.get('Right', {}).get('landmarks')
                right_score = hands_by_side.get('Right', {}).get('score', 0.0)
                left_landmarks = hands_by_side.get('Left', {}).get('landmarks')
                left_score = hands_by_side.get('Left', {}).get('score', 0.0)
                
                features = build_features_for_both_hands(right_landmarks, right_score, left_landmarks, left_score)
                
                # Hacer una predicción con ambas manos
                gesture_name, confidence, class_idx = predict_gesture(features)
                
                # Guardar datos esqueletales y predicción
                save_skeletal_data(current_timestamp, 
                                 right_landmarks=hands_by_side.get('Right', {}).get('landmarks'),
                                 right_score=hands_by_side.get('Right', {}).get('score', 0.0),
                                 left_landmarks=hands_by_side.get('Left', {}).get('landmarks'),
                                 left_score=hands_by_side.get('Left', {}).get('score', 0.0))
                
                log_prediction(current_timestamp, gesture_name, float(confidence), int(class_idx),
                             'Right' in hands_by_side, 'Left' in hands_by_side)
                
                # Imprimir en consola cada 30 frames
                if frame_count % 30 == 0:
                    print(f"[Frame {frame_count}] {gesture_name}: {confidence:.3f}")
                
                # Guardar frame cada FRAMES_SAVE_INTERVAL frames
                if frame_count % FRAMES_SAVE_INTERVAL == 0:
                    save_frame_with_skeleton(display_image, current_timestamp.split('T')[1], frame_count)
                
                # Dibujar resultados en todas las manos detectadas
                for hand_side, hand_data in hands_by_side.items():
                    landmarks = hand_data['landmarks']
                    
                    # Obtener posición del bounding box para mostrar el texto
                    x_coords = [lm.x * w for lm in landmarks]
                    y_coords = [lm.y * h for lm in landmarks]

                    x_min, x_max = int(min(x_coords)), int(max(x_coords))
                    y_min, y_max = int(min(y_coords)), int(max(y_coords))

                    # Agregar margen
                    margin = 10
                    x_min = max(0, x_min - margin)
                    y_min = max(0, y_min - margin)
                    x_max = min(w, x_max + margin)
                    y_max = min(h, y_max + margin)

                    # Espejar coordenadas X para dibujar sobre la imagen volteada
                    x_min_m = w - x_max
                    x_max_m = w - x_min

                    # Dibujar rectángulo alrededor de la mano
                    cv2.rectangle(display_image, (x_min_m, y_min), (x_max_m, y_max), (0, 255, 0), 2)

                    # Mostrar gesto y confianza
                    text_gesture = f"{gesture_name}"
                    text_confidence = f"Conf: {confidence:.2f}"

                    cv2.putText(display_image, f"{hand_side}: {text_gesture}",
                               (x_min_m, y_min - 30), cv2.FONT_HERSHEY_SIMPLEX,
                               0.7, (0, 255, 0), 2)
                    cv2.putText(display_image, text_confidence,
                               (x_min_m, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX,
                               0.6, (0, 255, 0), 1)

        cv2.imshow('Detección de Gestos en Directo', display_image)

        key = cv2.waitKey(5) & 0xFF
        if key == 27:  # ESC 
            print("Saliendo...")
            break
finally:
    if hand_detector.get('mode') == 'tasks':
        try:
            hand_detector['landmarker'].close()
        except Exception:
            pass

    cap.release()
    cv2.destroyAllWindows()
    print("Video cerrado")