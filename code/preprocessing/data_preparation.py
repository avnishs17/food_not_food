"""
Data preparation utilities for the Food/Not Food classification project.
"""

import os
import shutil
import random
import json
import tensorflow as tf
from nltk.corpus import wordnet as wn
import sys

# Add the parent directory to the path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def create_directory_structure():
    """Create the required directory structure for the project."""
    os.makedirs(config.FOOD_TRAIN_DIR, exist_ok=True)
    os.makedirs(config.NOT_FOOD_TRAIN_DIR, exist_ok=True)
    os.makedirs(config.FOOD_TEST_DIR, exist_ok=True)
    os.makedirs(config.NOT_FOOD_TEST_DIR, exist_ok=True)
    print(f"Created directory structure in {config.DATA_DIR}")


def copy_images(source_dir, dest_dir, split_ratio, num_images_per_class, num_classes, label):
    """
    Copy images from source directory to destination directory with the given split ratio.
    
    Args:
        source_dir (str): Source directory containing class folders with images
        dest_dir (str): Destination base directory
        split_ratio (float): Ratio of train/test split (e.g., 0.8 for 80% train, 20% test)
        num_images_per_class (int): Number of images to use per class
        num_classes (int): Number of classes to use
        label (str): Label for the images ('food' or 'not_food')
    """
    if not os.path.exists(source_dir):
        print(f"Source directory {source_dir} does not exist. Skipping.")
        return
        
    class_names = os.listdir(source_dir)
    random.shuffle(class_names)
    
    selected_classes = class_names[:num_classes]  # Select a specific number of classes
    
    for class_name in selected_classes:
        class_path = os.path.join(source_dir, class_name)
        if os.path.isdir(class_path):
            images = os.listdir(class_path)
            random.shuffle(images)
            
            # Limit to num_images_per_class
            images = images[:num_images_per_class]

            split_index = int(len(images) * split_ratio)
            train_images = images[:split_index]
            test_images = images[split_index:]

            # Copy train images
            train_class_dir = os.path.join(dest_dir, 'train', label, class_name)
            os.makedirs(train_class_dir, exist_ok=True)
            for img in train_images:
                try:
                    shutil.copy(os.path.join(class_path, img), train_class_dir)
                    print(f"Copied {os.path.join(class_path, img)} to {train_class_dir}")
                except PermissionError as e:
                    print(f"Permission denied: {e}")

            # Copy test images
            test_class_dir = os.path.join(dest_dir, 'test', label, class_name)
            os.makedirs(test_class_dir, exist_ok=True)
            for img in test_images:
                try:
                    shutil.copy(os.path.join(class_path, img), test_class_dir)
                    print(f"Copied {os.path.join(class_path, img)} to {test_class_dir}")
                except PermissionError as e:
                    print(f"Permission denied: {e}")


def get_food_list():
    """
    Get a list of food items from WordNet.
    
    Returns:
        list: List of food items
    """
    food = wn.synset('food.n.02')
    food_list = list(set([w for s in food.closure(lambda s:s.hyponyms()) for w in s.lemma_names()]))
    
    # Remove punctuation and lowercase
    food_list = [food_item.lower().replace('_', ' ') for food_item in food_list]
    
    return food_list


def load_class_mappings():
    """
    Load food and non-food class mappings from JSON files.
    
    Returns:
        tuple: (food_classes, non_food_classes) dictionaries
    """
    with open(config.FOOD_CLASSES_JSON, 'r') as f:
        food_classes = json.load(f)
    
    with open(config.NON_FOOD_CLASSES_JSON, 'r') as f:
        non_food_classes = json.load(f)
    
    # Convert keys to integers
    food_classes = {int(k): v for k, v in food_classes.items()}
    non_food_classes = {int(k): v for k, v in non_food_classes.items()}
    
    return food_classes, non_food_classes


def create_tf_datasets():
    """
    Create TensorFlow datasets for training and validation.
    
    Returns:
        tuple: (train_dataset, val_dataset) TensorFlow datasets
    """
    # Data preparation
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        config.TRAIN_DIR,
        seed=42,
        image_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        label_mode='categorical'  # Specify categorical labels
    )
    
    val_dataset = tf.keras.utils.image_dataset_from_directory(
        config.TEST_DIR,
        seed=42,
        image_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        label_mode='categorical'  # Specify categorical labels
    )
    
    # Data augmentation
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip(config.RANDOM_FLIP),
        tf.keras.layers.RandomRotation(config.RANDOM_ROTATION),
        tf.keras.layers.RandomZoom(config.RANDOM_ZOOM),
    ])
    
    # Apply data augmentation to the training dataset
    train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x), y))
    
    # Prefetch data for better performance
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return train_dataset, val_dataset


if __name__ == "__main__":
    # Example usage
    create_directory_structure()
    
    # Define source directories
    food_dirs = ['food-101/images', 'imagenet_food_images']
    not_food_dir = 'imagenet_not_food_images'
    
    # Process food directories
    for food_dir in food_dirs:
        copy_images(
            food_dir, 
            config.DATA_DIR, 
            config.SPLIT_RATIO, 
            config.NUM_IMAGES_PER_CLASS, 
            config.NUM_CLASSES_TO_USE, 
            'food'
        )
    
    # Process non-food directory
    copy_images(
        not_food_dir, 
        config.DATA_DIR, 
        config.SPLIT_RATIO, 
        config.NUM_IMAGES_PER_CLASS, 
        config.NUM_CLASSES_TO_USE, 
        'not_food'
    )
