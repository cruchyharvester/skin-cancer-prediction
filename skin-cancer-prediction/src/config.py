# part1_config_and_imports.py

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input, TimeDistributed, LSTM, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# ✅ Configuration
IMG_SIZE = 224
SEQUENCE_LENGTH = 4
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-4

# ✅ Dataset paths
DATASET_PATHS = [
    r"C:\Users\mahir\OneDrive\Documents\skin C\HAM10000_images_part_1",
    r"C:\Users\mahir\OneDrive\Documents\skin C\HAM10000_images_part_2"
]
