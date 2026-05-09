#DEEP LEARNING PARA DETECTAR GESTOS CON INFORMACIÓN SKELETAL

import tensorflow as tf
from tensorflow.python.keras import layers, models
import numpy as np
import time
import pathlib
import cv2

# 1. Cargar y preparar los datos 

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
DATASET_PATH = str(BASE_DIR / 'data' / 'dataset_skeletal_hand_gestures')

with open(DATASET_PATH + '/train.csv') as f:
    lines = f.readlines()