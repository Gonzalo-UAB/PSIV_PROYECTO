#DEEP LEARNING PARA DETECTAR GESTOS CON INFORMACIÓN SKELETAL

import os
import tensorflow as tf
from tensorflow.keras import layers, models
import pathlib
import xmltodict
import numpy as np
import pandas as pd
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Paths y constantes
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
DATASET_PATH = str(BASE_DIR / 'data' / 'dataset_skeletal_hand_gesture' / 'skeletal')

types_archives = ["test_gesture","test_pose", "train_pose"]

# Mapeo de gestos a índices
GESTURE_LABELS = {
    '02_l': 0, '04_fist_moved': 1, '06_index': 2, '07_ok': 3, '09_c': 4,
    '11_heavy': 5, '12_hang': 6, '13_two': 7, '14_three': 8, '15_four': 9,
    '16_five': 10, '19_down': 11, '20_left': 12, '21_right': 13, '22_up': 14
}

# Mapeo de poses a índices
POSE_LABELS = {
    '01_palm': 0, '02_l': 1, '04_fist_moved': 2, '05_down': 3, '06_index': 4,
    '07_ok': 5, '08_palm_m': 6, '09_c': 7, '11_heavy': 8, '12_hang': 9,
    '13_two': 10, '14_three': 11, '15_four': 12, '16_five': 13, '17_palm_u': 14,
    '18_up': 15
}

CACHE_DIR = BASE_DIR / 'cache'
CACHE_FILE = CACHE_DIR / 'skeletal_dataset.pkl'
FIGS_DIR = BASE_DIR / 'figs'


def extract_hand_data(xml_file_path):
    """Extrae datos de mano de archivo XML"""
    try:
        with open(xml_file_path, 'r') as f:
            data = xmltodict.parse(f.read())
        
        frame = data['opencv_storage']['Frame']
        frame_id = frame['ID']
        
        # Extraer features de ambas imágenes
        features = []

        for image_key in ['RigthImage', 'LeftImage']:
            if image_key not in frame['Images']:
                continue

            image = frame['Images'][image_key]
            if 'Hands' not in image or 'Right' not in image['Hands']:
                continue

            hands = image['Hands']['Right']

            if isinstance(hands, list):
                hands = hands[0]

            # Extraer 3D skeleton points
            skeleton_points = []

            # Centro de la mano
            if 'Center' in hands:
                center = [float(x) for x in hands['Center']['data'].strip().split()]
                skeleton_points.extend(center)
            else:
                skeleton_points.extend([0, 0, 0])

            # Muñeca - OPCIONAL (algunos XMLs no lo tienen)
            if 'WristPosition' in hands:
                wrist = [float(x) for x in hands['WristPosition']['data'].strip().split()]
                skeleton_points.extend(wrist)
            else:
                skeleton_points.extend([0, 0, 0])

            # Datos de dedos
            if 'Fingers' in hands:
                for finger_name in ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']:
                    if finger_name in hands['Fingers']:
                        finger = hands['Fingers'][finger_name]

                        # TipPosition
                        if 'TipPosition' in finger:
                            tip = [float(x) for x in finger['TipPosition']['data'].strip().split()]
                            skeleton_points.extend(tip)
                        else:
                            skeleton_points.extend([0, 0, 0])

                        # MCPPosition
                        if 'mcpPosition' in finger:
                            mcp = [float(x) for x in finger['mcpPosition']['data'].strip().split()]
                            skeleton_points.extend(mcp)
                        else:
                            skeleton_points.extend([0, 0, 0])
                    else:
                        # Si el dedo no existe, añadir ceros
                        skeleton_points.extend([0, 0, 0, 0, 0, 0])
            else:
                # Si no hay dedos, añadir ceros para todos
                skeleton_points.extend([0] * 30)  # 5 dedos * 6 valores (tip + mcp)

            # Confidence
            confidence = float(hands.get('Confidence', 0))

            # GrabStrength y PinchStrength
            grab_strength = float(hands.get('GrabStrength', 0))
            pinch_strength = float(hands.get('PinchStrength', 0))

            # isExtended para cada dedo
            extended = []
            if 'Fingers' in hands:
                for finger_name in ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']:
                    if finger_name in hands['Fingers']:
                        extended.append(int(hands['Fingers'][finger_name].get('isExtended', 0)))
                    else:
                        extended.append(0)
            else:
                extended = [0] * 5

            # Combinar todas las features
            image_features = skeleton_points + [confidence, grab_strength, pinch_strength] + extended
            features.extend(image_features)

            # Si no hay features válidas, retornar None
        if len(features) == 0:
            return None

        return np.array(features, dtype=np.float32)

    except Exception as e:
        print(f"Error procesando {xml_file_path}: {e}")
        return None


def collect_dataset_files():
    """Recolecta todos los XML y su etiqueta una sola vez."""
    samples = []

    for i in range(15):
        patient_id = f'0{i}' if i < 10 else str(i)
        patient_path = os.path.join(DATASET_PATH, patient_id)

        for archive_type in types_archives:
            archive_path = os.path.join(patient_path, archive_type)

            if not os.path.exists(archive_path):
                continue

            gesture_folders = [f for f in os.listdir(archive_path)
                               if os.path.isdir(os.path.join(archive_path, f))]

            for gesture_folder in gesture_folders:
                gesture_path = os.path.join(archive_path, gesture_folder)
                label = extract_label_from_path(gesture_path, archive_type)

                if label == -1:
                    continue

                for root, _, files in os.walk(gesture_path):
                    for file in files:
                        if file.endswith('.xml'):
                            samples.append((os.path.join(root, file), label, patient_id, archive_type, gesture_folder))

    return samples


def load_cache():
    if not CACHE_FILE.exists():
        return None

    with open(CACHE_FILE, 'rb') as f:
        return pickle.load(f)


def save_cache(dataset):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)


def _extract_sample(sample):
    xml_file, label, patient_id, archive_type, gesture_folder = sample
    features = extract_hand_data(xml_file)

    if features is None:
        return None

    return features, label, {
        'patient': patient_id,
        'archive_type': archive_type,
        'gesture': gesture_folder,
        'xml_file': xml_file
    }


def extract_label_from_path(folder_path, archive_type):
    """Extrae la etiqueta (gesto/pose) del path"""
    folder_name = os.path.basename(folder_path)
    
    if 'gesture' in archive_type:
        return GESTURE_LABELS.get(folder_name, -1)
    else:  # pose
        return POSE_LABELS.get(folder_name, -1)


def build_dataset(force_rebuild=False, use_cache=True, max_workers=None):
    """Construye el dataset completo para deep learning"""
    if use_cache and not force_rebuild:
        cached_dataset = load_cache()
        if cached_dataset is not None:
            print(f"Dataset cargado desde caché: {CACHE_FILE}")
            return cached_dataset['X'], cached_dataset['y'], cached_dataset['metadata']

    print("Procesando dataset...")
    samples = collect_dataset_files()

    X = []
    y = []
    metadata = []

    worker_count = max_workers or max(1, (os.cpu_count() or 2) - 1)
    print(f"Extrayendo XML en paralelo con {worker_count} procesos...")

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(_extract_sample, sample) for sample in samples]

        for future in as_completed(futures):
            result = future.result()
            if result is None:
                continue

            features, label, sample_metadata = result
            X.append(features)
            y.append(label)
            metadata.append(sample_metadata)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    print(f"Dataset construido: {len(X)} muestras")
    print(f"Dimensión de features: {X.shape}")
    print(f"Clases: {len(np.unique(y))}")

    dataset = {'X': X, 'y': y, 'metadata': metadata}
    if use_cache:
        save_cache(dataset)
        print(f"Dataset serializado en: {CACHE_FILE}")

    return X, y, metadata


def normalize_data(X_train, X_test):
    """Normaliza los datos usando StandardScaler"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler


def create_model(input_dim, num_classes):
    """Crea un modelo de deep learning"""
    model = models.Sequential([
        layers.Dense(256, activation='relu', input_dim=input_dim),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def save_training_plots(history):
    """Guarda las curvas de entrenamiento en la carpeta figs."""
    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history.history['accuracy']) + 1)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.history['accuracy'], label='Train Accuracy')
    plt.plot(epochs, history.history['val_accuracy'], label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history['loss'], label='Train Loss')
    plt.plot(epochs, history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(FIGS_DIR / 'training_history.png', dpi=200, bbox_inches='tight')
    plt.close()


def save_confusion_matrix(model, X_test_scaled, y_test):
    """Calcula y guarda la matriz de confusión y el reporte de clasificación."""
    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    y_prob = model.predict(X_test_scaled, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)

    print("\nClassification report:\n")
    print(report)

    display = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(10, 10))
    display.plot(ax=ax, cmap='Blues', colorbar=False, values_format='d')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(FIGS_DIR / 'confusion_matrix.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    with open(FIGS_DIR / 'classification_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)

def main():
    """Función principal para entrenar el modelo"""
    
    # Construir dataset
    X, y, metadata = build_dataset()
    
    if len(X) == 0:
        print("No se pudieron cargar datos del dataset")
        return
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Normalizar datos
    X_train_scaled, X_test_scaled, scaler = normalize_data(X_train, X_test)
    
    # Crear modelo
    num_classes = len(np.unique(y))
    input_dim = X.shape[1]
    
    model = create_model(input_dim, num_classes)
    
    print("\nModelo:")
    model.summary()
    
    # Entrenar modelo
    print("\nEntrenando modelo...")
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_test_scaled, y_test),
        epochs=50,
        batch_size=32,
        verbose=1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
    )
    
    # Evaluación
    print("\nEvaluando modelo...")
    test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Guardar diagnósticos visuales
    save_training_plots(history)
    save_confusion_matrix(model, X_test_scaled, y_test)
    print(f"Figuras guardadas en: {FIGS_DIR}")
    
    # Guardar modelo
    model_path = BASE_DIR / 'models' / 'gesture_model.keras'
    model_path.parent.mkdir(exist_ok=True)
    model.save(str(model_path))
    print(f"\nModelo guardado en: {model_path}")
    
    # Guardar scaler
    import pickle
    scaler_path = BASE_DIR / 'models' / 'scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler guardado en: {scaler_path}")




if __name__ == "__main__":
    main()
    