"""
Configuration settings for the Food/Not Food classification project.
"""

import os

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Data directories
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
FOOD_TRAIN_DIR = os.path.join(TRAIN_DIR, 'food')
NOT_FOOD_TRAIN_DIR = os.path.join(TRAIN_DIR, 'not_food')
FOOD_TEST_DIR = os.path.join(TEST_DIR, 'food')
NOT_FOOD_TEST_DIR = os.path.join(TEST_DIR, 'not_food')

# Image processing settings
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 64

# Training settings
EPOCHS = 10
LEARNING_RATE = 0.001
SPLIT_RATIO = 0.85
NUM_CLASSES = 2  # Food and Not Food

# Data augmentation settings
RANDOM_ROTATION = 0.1
RANDOM_ZOOM = 0.1
RANDOM_FLIP = "horizontal"

# Model settings
MODEL_ARCHITECTURE = "EfficientNetV2B0"
MODEL_WEIGHTS = "imagenet"
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, 'food_not_food_model.h5')
TFLITE_MODEL_PATH = os.path.join(MODEL_DIR, 'food_not_food_model.tflite')

# Class mapping
CLASS_MAPPING = {
    0: "not_food",
    1: "food"
}

# JSON files
FOOD_CLASSES_JSON = os.path.join(BASE_DIR, 'research', 'food_classes.json')
NON_FOOD_CLASSES_JSON = os.path.join(BASE_DIR, 'research', 'non_food_classes.json')

# Data preprocessing settings
NUM_IMAGES_PER_CLASS = 50
NUM_CLASSES_TO_USE = 140
