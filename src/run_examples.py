"""
Ejemplos de uso para reconstruir videos de gestos.
"""

from reconstruct_skeleton_video import SkeletonVideoReconstructor
import os

# Ruta base del dataset
DATASET_BASE = "../data/dataset_skeletal_hand_gesture/skeletal"
OUTPUT_BASE = "../skeleton_videos"


def ejemplo_1_video_unico():
    """
    Ejemplo 1: Reconstruir un video de un gesto específico.
    """
    print("=" * 60)
    print("EJEMPLO 1: Reconstruir un video de un gesto específico")
    print("=" * 60)
    
    # Ruta a una secuencia específica
    gesture_path = os.path.join(
        DATASET_BASE,
        "00",  # Usuario 0
        "test_gesture",  # Tipo de gesto
        "02_l",  # Gesto: letra L
        "00"  # Secuencia 0
    )
    
    output_video = os.path.join(OUTPUT_BASE, "letra_L_usuario_0.mp4")
    
    # Crear reconstructor
    reconstructor = SkeletonVideoReconstructor(fps=30)
    
    # Reconstruir video
    success = reconstructor.reconstruct_video(
        gesture_path=gesture_path,
        output_video_path=output_video,
        gesture_label="Letra L - Usuario 0"
    )
    
    if success:
        print(f"✓ Video guardado: {output_video}")
    else:
        print(f"✗ Error al crear el video")


def ejemplo_2_todos_usuarios_un_gesto():
    """
    Ejemplo 2: Reconstruir todos los videos de un gesto específico
    para todos los usuarios.
    """
    print("\n" + "=" * 60)
    print("EJEMPLO 2: Todos los usuarios para un gesto (02_l)")
    print("=" * 60)
    
    reconstructor = SkeletonVideoReconstructor(fps=30)
    
    # Procesar para usuarios 0-4, gesto "02_l"
    for user_id in range(5):  # Usuarios 0-4
        user_str = str(user_id).zfill(2)
        
        gesture_type_path = os.path.join(
            DATASET_BASE,
            user_str,
            "test_gesture",
            "02_l"
        )
        
        if not os.path.exists(gesture_type_path):
            print(f"Carpeta no encontrada: {gesture_type_path}")
            continue
        
        # Procesar cada secuencia
        for seq_id in os.listdir(gesture_type_path):
            seq_path = os.path.join(gesture_type_path, seq_id)
            
            if not os.path.isdir(seq_path):
                continue
            
            output_dir = os.path.join(OUTPUT_BASE, f"usuario_{user_id}")
            os.makedirs(output_dir, exist_ok=True)
            
            output_video = os.path.join(output_dir, f"02_l_seq_{seq_id}.mp4")
            
            print(f"\nProcesando usuario {user_id}, secuencia {seq_id}...")
            
            reconstructor.reconstruct_video(
                gesture_path=seq_path,
                output_video_path=output_video,
                gesture_label=f"Letra L - Usuario {user_id} - Seq {seq_id}"
            )


def ejemplo_3_todos_gestos_usuario():
    """
    Ejemplo 3: Reconstruir todos los gestos de un usuario específico.
    """
    print("\n" + "=" * 60)
    print("EJEMPLO 3: Todos los gestos de un usuario (Usuario 0)")
    print("=" * 60)
    
    reconstructor = SkeletonVideoReconstructor(fps=30)
    
    user_id = 0
    user_str = str(user_id).zfill(2)
    gesture_type_path = os.path.join(DATASET_BASE, user_str, "test_gesture")
    
    if not os.path.exists(gesture_type_path):
        print(f"Carpeta no encontrada: {gesture_type_path}")
        return
    
    # Procesar cada gesto
    gesture_dirs = sorted(os.listdir(gesture_type_path))
    
    for gesture_name in gesture_dirs:
        gesture_path = os.path.join(gesture_type_path, gesture_name)
        
        if not os.path.isdir(gesture_path):
            continue
        
        print(f"\nProcesando gesto: {gesture_name}")
        
        # Procesar primera secuencia (00) de cada gesto
        seq_path = os.path.join(gesture_path, "00")
        
        if not os.path.exists(seq_path):
            print(f"  Secuencia 00 no encontrada para {gesture_name}")
            continue
        
        output_dir = os.path.join(OUTPUT_BASE, f"usuario_{user_id}_gestos")
        os.makedirs(output_dir, exist_ok=True)
        
        output_video = os.path.join(output_dir, f"{gesture_name}.mp4")
        
        reconstructor.reconstruct_video(
            gesture_path=seq_path,
            output_video_path=output_video,
            gesture_label=f"{gesture_name} - Usuario {user_id}"
        )


def ejemplo_4_batch_completo():
    """
    Ejemplo 4: Procesamiento en batch de múltiples usuarios y gestos.
    """
    print("\n" + "=" * 60)
    print("EJEMPLO 4: Procesamiento en batch (2 usuarios, test_gesture)")
    print("=" * 60)
    
    reconstructor = SkeletonVideoReconstructor(fps=30)
    
    # Procesar usuarios 0 y 1, solo test_gesture
    reconstructor.batch_reconstruct(
        dataset_path=DATASET_BASE,
        output_dir=os.path.join(OUTPUT_BASE, "batch_completo"),
        user_ids=[0, 1],  # Solo usuarios 0 y 1
        gesture_types=['test_gesture']  # Solo gestos de prueba
    )


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        ejemplo_num = sys.argv[1]
        
        if ejemplo_num == '1':
            ejemplo_1_video_unico()
        elif ejemplo_num == '2':
            ejemplo_2_todos_usuarios_un_gesto()
        elif ejemplo_num == '3':
            ejemplo_3_todos_gestos_usuario()
        elif ejemplo_num == '4':
            ejemplo_4_batch_completo()
        else:
            print(f"Ejemplo {ejemplo_num} no encontrado")
            print("\nUsos:")
            print("  python run_examples.py 1  - Video único")
            print("  python run_examples.py 2  - Todos usuarios un gesto")
            print("  python run_examples.py 3  - Todos gestos de un usuario")
            print("  python run_examples.py 4  - Batch completo")
    else:
        print("=" * 60)
        print("EJEMPLOS DE RECONSTRUCCIÓN DE VIDEOS DE GESTOS")
        print("=" * 60)
        print("\nUsos:")
        print("  python run_examples.py 1  - Video único")
        print("  python run_examples.py 2  - Todos usuarios un gesto")
        print("  python run_examples.py 3  - Todos gestos de un usuario")
        print("  python run_examples.py 4  - Batch completo")
        print("\nEjemplo:")
        print("  python run_examples.py 1")
