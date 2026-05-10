#DEEP LEARNING PARA DETECTAR GESTOS CON INFORMACIÓN SKELETAL

import os

import tensorflow as tf
from tensorflow.python.keras import layers, models
import numpy as np
import time
import pathlib
import cv2

# 1. Cargar y preparar los datos 

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
DATASET_PATH = str(BASE_DIR / 'data' / 'dataset_skeletal_hand_gestures')

types_archives = ["test_gesture","test_pose", "train_pose"]

#Recorrer los 15 pacientes 
for i in range(15):
    if i < 9:
        patient_path = DATASET_PATH + f'/0{i}'
    else:
        patient_path = DATASET_PATH + f'/{i}'

    #Recorrer las 3 carpetas de cada paciente (train_pose, test_gesture, test_pose)
    for j in types_archives:
        archive_path = patient_path + f'/{j}'

        #Encontrar las subcarpetas 
        subfolders = [f.path for f in os.scandir(archive_path) if f.is_dir()]
        #Recorrer las 15 gestos de cada gesto
        for k in subfolders:
            gesture_path = archive_path + f'/{k+1}.png'

            #Encontrar las imagenes de cada gesto
            
            for l in range(20):  # Assuming 20 images per gesture

with open(DATASET_PATH + '/train.csv') as f:
    lines = f.readlines()


