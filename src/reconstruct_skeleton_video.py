"""
Script para reconstruir videos de gestos a partir de datos esqueletales XML.
Dibuja el esqueleto de la mano en cada frame y los compila en un video.
"""

import xml.etree.ElementTree as ET
import cv2
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Tuple
import argparse


class SkeletonVideoReconstructor:
    """Reconstruye videos de gestos a partir de frames con información esqueletal."""
    
    # Mapeo de nombres de dedos y articulaciones
    FINGER_NAMES = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
    JOINT_NAMES = ['TipPosition', 'dipPosition', 'pipPosition', 'mcpPosition']
    
    # Colores para cada dedo (BGR)
    FINGER_COLORS = {
        'Thumb': (255, 0, 0),      # Azul
        'Index': (0, 255, 0),      # Verde
        'Middle': (0, 0, 255),     # Rojo
        'Ring': (255, 255, 0),     # Cian
        'Pinky': (255, 0, 255)     # Magenta
    }
    
    # Conexiones entre articulaciones para dibujar líneas
    FINGER_SKELETON = [
        'TipPosition', 'dipPosition', 'pipPosition', 'mcpPosition'
    ]
    
    def __init__(self, output_size: Tuple[int, int] = (800, 600), fps: int = 30):
        """
        Inicializa el reconstructor.
        
        Args:
            output_size: Tamaño del video de salida (ancho, alto)
            fps: Fotogramas por segundo del video
        """
        self.output_size = output_size
        self.fps = fps
        
    def parse_xml_frame(self, xml_path: str) -> Dict:
        """
        Parsea un archivo XML de frame y extrae posiciones de puntos clave.
        
        Args:
            xml_path: Ruta al archivo XML
            
        Returns:
            Diccionario con información del esqueleto
        """
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            frame_data = {
                'hand_center': None,
                'fingers': {}
            }
            
            # Buscar la mano (RigthImage/Right)
            right_image = root.find('.//RigthImage')
            if right_image is None:
                return frame_data
                
            hands = right_image.find('.//Hands')
            if hands is None:
                return frame_data
            
            hand = hands.find('.//Right')
            if hand is None:
                return frame_data
            
            # Extraer centro de la mano
            center_elem = hand.find('Center')
            if center_elem is not None and center_elem.find('data') is not None:
                center_data = center_elem.find('data').text.strip().split()
                frame_data['hand_center'] = tuple(float(x) for x in center_data)
            
            # Extraer información de dedos
            fingers_elem = hand.find('Fingers')
            if fingers_elem is not None:
                for finger_name in self.FINGER_NAMES:
                    finger = fingers_elem.find(finger_name)
                    if finger is not None:
                        frame_data['fingers'][finger_name] = {}
                        
                        # Extraer posiciones de articulaciones
                        for joint_name in self.JOINT_NAMES:
                            joint = finger.find(joint_name)
                            if joint is not None and joint.find('data') is not None:
                                joint_data = joint.find('data').text.strip().split()
                                frame_data['fingers'][finger_name][joint_name] = tuple(
                                    float(x) for x in joint_data
                                )
                        
                        # Extraer si está extendido
                        is_extended = finger.find('isExtended')
                        if is_extended is not None:
                            frame_data['fingers'][finger_name]['isExtended'] = (
                                is_extended.text == '1'
                            )
            
            return frame_data
            
        except Exception as e:
            print(f"Error parseando {xml_path}: {e}")
            return {}
    
    def project_3d_to_2d(self, point_3d: Tuple, canvas_size: Tuple[int, int]) -> Tuple[int, int]:
        """
        Proyecta un punto 3D a 2D para renderizar en la imagen.
        Usa una proyección ortográfica con escalado y centrado.
        
        Args:
            point_3d: Punto en 3D (x, y, z)
            canvas_size: Tamaño del canvas (ancho, alto)
            
        Returns:
            Coordenadas 2D (x, y)
        """
        if len(point_3d) < 3:
            return (int(canvas_size[0]/2), int(canvas_size[1]/2))
        
        x, y, z = point_3d[:3]
        
        # Proyección simple: usar X y Y, ignorar Z
        # Escalar a rango de canvas con margen
        margin = 100
        
        # Asumir que el rango aproximado es [-200, 200] para X y Y
        scale = (canvas_size[0] - 2*margin) / 400
        
        canvas_x = int(canvas_size[0]/2 + x * scale)
        canvas_y = int(canvas_size[1]/2 - y * scale)  # Invertir Y para que sea normal
        
        # Clamp a los límites del canvas
        canvas_x = max(0, min(canvas_size[0]-1, canvas_x))
        canvas_y = max(0, min(canvas_size[1]-1, canvas_y))
        
        return (canvas_x, canvas_y)
    
    def draw_skeleton(self, frame: np.ndarray, skeleton_data: Dict) -> np.ndarray:
        """
        Dibuja el esqueleto de la mano en el frame.
        
        Args:
            frame: Frame en blanco para dibujar
            skeleton_data: Datos del esqueleto
            
        Returns:
            Frame con el esqueleto dibujado
        """
        if not skeleton_data or not skeleton_data.get('fingers'):
            return frame
        
        # Dibujar cada dedo
        for finger_name, finger_data in skeleton_data['fingers'].items():
            if not finger_data or not isinstance(finger_data, dict):
                continue
            
            color = self.FINGER_COLORS.get(finger_name, (200, 200, 200))
            
            # Obtener posiciones de articulaciones en orden
            joints_2d = []
            for joint_name in self.JOINT_NAMES:
                if joint_name in finger_data:
                    pos_3d = finger_data[joint_name]
                    pos_2d = self.project_3d_to_2d(pos_3d, self.output_size)
                    joints_2d.append(pos_2d)
            
            # Dibujar líneas conectando articulaciones
            for i in range(len(joints_2d) - 1):
                pt1 = joints_2d[i]
                pt2 = joints_2d[i + 1]
                cv2.line(frame, pt1, pt2, color, 2)
            
            # Dibujar círculos en cada articulación
            for pt in joints_2d:
                cv2.circle(frame, pt, 4, color, -1)
        
        # Dibujar centro de la mano si está disponible
        if skeleton_data.get('hand_center'):
            center_2d = self.project_3d_to_2d(skeleton_data['hand_center'], self.output_size)
            cv2.circle(frame, center_2d, 6, (200, 200, 200), -1)
            cv2.circle(frame, center_2d, 6, (100, 100, 100), 2)
        
        return frame
    
    def get_frame_files(self, gesture_path: str) -> List[str]:
        """
        Obtiene lista ordenada de archivos XML de frames.
        
        Args:
            gesture_path: Ruta a la carpeta de secuencia de gesto
            
        Returns:
            Lista ordenada de rutas a archivos XML
        """
        xml_files = sorted([
            os.path.join(gesture_path, f) 
            for f in os.listdir(gesture_path) 
            if f.endswith('.xml')
        ])
        return xml_files
    
    def reconstruct_video(self, gesture_path: str, output_video_path: str, 
                         gesture_label: str = "") -> bool:
        """
        Reconstruye un video a partir de frames con datos esqueletales.
        
        Args:
            gesture_path: Ruta a la carpeta con frames XML
            output_video_path: Ruta del video de salida
            gesture_label: Etiqueta del gesto a mostrar en el video
            
        Returns:
            True si fue exitoso, False en caso contrario
        """
        # Obtener frames
        frame_files = self.get_frame_files(gesture_path)
        
        if not frame_files:
            print(f"No se encontraron archivos XML en {gesture_path}")
            return False
        
        print(f"Encontrados {len(frame_files)} frames en {gesture_path}")
        
        # Crear escritor de video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(
            output_video_path, 
            fourcc, 
            self.fps, 
            self.output_size
        )
        
        if not writer.isOpened():
            print(f"Error al crear el video {output_video_path}")
            return False
        
        # Procesar cada frame
        for i, frame_file in enumerate(frame_files):
            # Parse XML
            skeleton_data = self.parse_xml_frame(frame_file)
            
            # Crear canvas blanco
            canvas = np.ones(
                (self.output_size[1], self.output_size[0], 3), 
                dtype=np.uint8
            ) * 255
            
            # Dibujar esqueleto
            canvas = self.draw_skeleton(canvas, skeleton_data)
            
            # Agregar información de texto
            text = f"Frame {i+1}/{len(frame_files)}"
            if gesture_label:
                text += f" - {gesture_label}"
            
            cv2.putText(
                canvas, 
                text, 
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2
            )
            
            # Escribir frame al video
            writer.write(canvas)
            
            if (i + 1) % max(1, len(frame_files) // 10) == 0:
                print(f"  Procesados {i+1}/{len(frame_files)} frames")
        
        writer.release()
        print(f"Video guardado en {output_video_path}")
        return True
    
    def batch_reconstruct(self, dataset_path: str, output_dir: str, 
                         user_ids: List[int] = None, 
                         gesture_types: List[str] = None) -> None:
        """
        Reconstruye múltiples videos en batch.
        
        Args:
            dataset_path: Ruta base del dataset (carpeta skeletal)
            output_dir: Directorio para guardar los videos
            user_ids: Lista de IDs de usuario a procesar (None = todos)
            gesture_types: Tipos de gesto a procesar: 'test_gesture', 'test_pose', etc.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        dataset_path = Path(dataset_path)
        
        # Obtener lista de usuarios
        user_dirs = sorted([d for d in dataset_path.iterdir() if d.is_dir()])
        
        if user_ids:
            user_dirs = [d for d in user_dirs if int(d.name) in user_ids]
        
        total = 0
        processed = 0
        
        for user_dir in user_dirs:
            user_id = user_dir.name
            
            for gesture_type_dir in user_dir.iterdir():
                if not gesture_type_dir.is_dir():
                    continue
                
                gesture_type = gesture_type_dir.name
                
                if gesture_types and gesture_type not in gesture_types:
                    continue
                
                # Buscar gestos específicos
                for gesture_dir in gesture_type_dir.iterdir():
                    if not gesture_dir.is_dir():
                        continue
                    
                    gesture_name = gesture_dir.name
                    
                    # Buscar secuencias
                    for seq_dir in gesture_dir.iterdir():
                        if not seq_dir.is_dir():
                            continue
                        
                        seq_id = seq_dir.name
                        total += 1
                        
                        # Crear ruta de salida
                        output_subdir = os.path.join(
                            output_dir,
                            user_id,
                            gesture_type,
                            gesture_name
                        )
                        os.makedirs(output_subdir, exist_ok=True)
                        
                        output_video = os.path.join(
                            output_subdir,
                            f"sequence_{seq_id}.mp4"
                        )
                        
                        print(f"\n[{total}] Procesando: {user_id}/{gesture_type}/{gesture_name}/{seq_id}")
                        
                        label = f"{gesture_name} - Seq {seq_id}"
                        success = self.reconstruct_video(
                            str(seq_dir),
                            output_video,
                            label
                        )
                        
                        if success:
                            processed += 1
        
        print(f"\n\nResumen: {processed}/{total} videos procesados exitosamente")


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruir videos de gestos desde datos esqueletales XML"
    )
    parser.add_argument(
        '--gesture_path',
        type=str,
        help='Ruta a una carpeta de secuencia específica para un solo video'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='gesture_video.mp4',
        help='Ruta del video de salida'
    )
    parser.add_argument(
        '--label',
        type=str,
        default='',
        help='Etiqueta del gesto a mostrar en el video'
    )
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Procesar múltiples videos en batch'
    )
    parser.add_argument(
        '--dataset_path',
        type=str,
        help='Ruta base del dataset para procesamiento en batch'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='skeleton_videos',
        help='Directorio para guardar los videos en batch'
    )
    parser.add_argument(
        '--user_ids',
        type=int,
        nargs='+',
        help='IDs de usuarios a procesar (ej: 0 1 2)'
    )
    parser.add_argument(
        '--gesture_types',
        type=str,
        nargs='+',
        help='Tipos de gesto a procesar (ej: test_gesture test_pose)'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='Fotogramas por segundo del video de salida'
    )
    
    args = parser.parse_args()
    
    reconstructor = SkeletonVideoReconstructor(fps=args.fps)
    
    if args.batch:
        if not args.dataset_path:
            print("Error: se requiere --dataset_path para procesamiento en batch")
            return
        
        reconstructor.batch_reconstruct(
            args.dataset_path,
            args.output_dir,
            user_ids=args.user_ids,
            gesture_types=args.gesture_types
        )
    else:
        if not args.gesture_path:
            print("Error: se requiere --gesture_path para un video individual")
            return
        
        reconstructor.reconstruct_video(
            args.gesture_path,
            args.output,
            args.label
        )


if __name__ == '__main__':
    main()
