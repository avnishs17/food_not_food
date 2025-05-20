"""
Download and preprocess data for the Food/Not Food classification project.

This script:
1. Downloads ImageNet data using the Hugging Face datasets library
2. Filters the data into food and non-food classes
3. Saves the images to appropriate directories in the data folder

Usage:
    python src/classifier/utils/download_and_preprocess.py

The script will create:
- data/ - Main data directory in the project root
  ├── food/ - Directory containing food images
  └── not_food/ - Directory containing non-food images

Note: This script should be run from the project root directory to ensure
      the data is saved in the correct location.
"""

import os
import sys
import json
import random
import argparse
from tqdm import tqdm
from PIL import Image
from nltk.corpus import wordnet as wn
from datasets import load_dataset
import nltk
import pathlib
import re

# you need to run huggingface-cli login before this to set your HF_TOKEN
# Ensure required NLTK data is downloaded
nltk.download('wordnet', quiet=True)

# Get the project root directory
def get_project_root():
    """Get the absolute path to the project root directory."""
    # When running from src/classifier/utils, we need to go up 3 levels
    current_file = pathlib.Path(__file__).resolve()
    return str(current_file.parent.parent.parent.parent)

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

def filter_imagenet_classes(imagenet_classes, food_list):
    """
    Filter ImageNet classes into food and non-food categories.

    Args:
        imagenet_classes (list): List of ImageNet class names
        food_list (list): List of food items from WordNet

    Returns:
        tuple: (food_classes, non_food_classes) dictionaries
    """
    # Create dictionaries to store food and non-food classes
    food_classes = {}
    non_food_classes = {}

    # Filter classes based on food list
    for i, name in enumerate(imagenet_classes):
        if any(food in name.lower() for food in food_list):
            food_classes[i] = name
        else:
            non_food_classes[i] = name

    # Manually remove non-food items from food classes
    not_food_ids = [
        1, 2, 4, 5, 8, 12, 13, 14, 18, 25, 32, 40, 46, 47, 49, 50, 52, 53, 54, 55,
        64, 68, 70, 80, 82, 83, 84, 85, 86, 92, 94, 99, 102, 103, 107, 109, 111, 113,
        117, 118, 119, 120, 121, 122, 123, 124, 125, 136, 137, 141, 143, 147, 153,
        161, 169, 190, 200, 209, 224, 227, 229, 230, 235, 238, 239, 245, 248, 251,
        252, 254, 256, 275, 298, 300, 301, 302, 303, 304, 305, 306, 313, 322, 323,
        324, 325, 326, 327, 329, 330, 331, 332, 333, 334, 339, 346, 376, 386, 390,
        391, 393, 395, 396, 397, 428, 431, 432, 445, 448, 461, 467, 471, 476, 477,
        479, 480, 498, 499, 503, 513, 520, 534, 537, 542, 580, 586, 587, 588, 598,
        607, 613, 619, 629, 630, 632, 644, 678, 679, 684, 694, 713, 718, 723, 726,
        729, 731, 739, 750, 752, 758, 762, 773, 778, 786, 798, 811, 821, 822, 829,
        837, 854, 856, 859, 874, 886, 891, 897, 905, 921, 923, 973, 975, 991, 996, 998
    ]

    # Remove non-food items from food classes
    to_remove = [key for key in not_food_ids if key in food_classes]
    for key in to_remove:
        del food_classes[key]

    # Manually add food items that were incorrectly classified as non-food
    keys_to_add = [928, 929, 961, 965, 966, 967, 969, 987]
    for key in keys_to_add:
        if key in non_food_classes:
            food_classes[key] = non_food_classes[key]
            del non_food_classes[key]

    return food_classes, non_food_classes

def preprocess_and_save_images(dataset, selected_classes, output_dir, images_per_class, classes_dict):
    """
    Preprocess and save images from the dataset.

    Args:
        dataset: Dataset containing images
        selected_classes (list): List of class IDs to include
        output_dir (str): Directory to save images to
        images_per_class (int): Number of images to save per class
        classes_dict (dict): Dictionary mapping class IDs to class names
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize counter for each class
    class_image_counts = {class_id: 0 for class_id in selected_classes}

    # Calculate total images to process
    total_images = len(selected_classes) * images_per_class

    # Create progress bar
    progress_bar = tqdm(total=total_images, desc=f"Saving images to {output_dir}", unit="img",dynamic_ncols=True)

    def save_image(image, class_id):
        """Save an image to the output directory."""
        count = class_image_counts[class_id]
        class_name = classes_dict[class_id]
        # Replace forbidden character for filename

        safe_class_name = re.sub(r"[^\w]", "_", class_name)
        safe_class_name = re.sub(r"__+", "_", safe_class_name).strip("_")
        img_filename = f"{safe_class_name}_{count}.jpg"
        img_path = os.path.join(output_dir, img_filename)

        # Convert image to RGB if it is grayscale or another mode
        if image.mode != 'RGB':
            image = image.convert("RGB")

        # Save the image
        image.save(img_path)
        class_image_counts[class_id] += 1
        progress_bar.update(1)

    # Process each example in the dataset
    for example in dataset:
        label = example['label']
        if label in selected_classes and class_image_counts[label] < images_per_class:
            image = example['image']
            save_image(image, label)

        # Check if we have enough images for all classes
        if all(count >= images_per_class for count in class_image_counts.values()):
            break

    progress_bar.close()
    print(f"Saved {sum(class_image_counts.values())} images to {output_dir}")

def save_class_mappings(food_classes, non_food_classes, output_dir):
    """
    Save class mappings to JSON files.

    Args:
        food_classes (dict): Dictionary of food classes
        non_food_classes (dict): Dictionary of non-food classes
        output_dir (str): Directory to save JSON files to
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save food classes
    food_json_path = os.path.join(output_dir, 'food_classes.json')
    with open(food_json_path, 'w') as f:
        json.dump(food_classes, f, indent=4)

    # Save non-food classes
    non_food_json_path = os.path.join(output_dir, 'non_food_classes.json')
    with open(non_food_json_path, 'w') as f:
        json.dump(non_food_classes, f, indent=4)

    print(f"Saved class mappings to {output_dir}")

def main(args):
    """Main function to download and preprocess data."""
    # Get the project root directory
    project_root = get_project_root()

    # Create absolute paths for output directories
    data_dir = os.path.join(project_root, args.data_dir)
    food_dir = os.path.join(data_dir, args.food_subdir)
    not_food_dir = os.path.join(data_dir, args.not_food_subdir)

    # Create output directories
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(food_dir, exist_ok=True)
    os.makedirs(not_food_dir, exist_ok=True)

    print(f"Data will be saved to:")
    print(f"  - Main data directory: {data_dir}")
    print(f"  - Food images: {food_dir}")
    print(f"  - Non-food images: {not_food_dir}")

    # Load ImageNet dataset
    print("Loading ImageNet dataset...")
    dataset = load_dataset("imagenet-1k", split="train", streaming=True, trust_remote_code=True)

    # Get list of ImageNet classes
    imagenet_classes = dataset.features['label'].names

    # Get food list from WordNet
    print("Getting food list from WordNet...")
    food_list = get_food_list()

    # Filter ImageNet classes
    print("Filtering ImageNet classes...")
    food_classes, non_food_classes = filter_imagenet_classes(imagenet_classes, food_list)

    print(f"Found {len(food_classes)} food classes and {len(non_food_classes)} non-food classes")

    # Save class mappings
    save_class_mappings(food_classes, non_food_classes, data_dir)

    # Randomly select classes
    food_selected_classes = random.sample(list(food_classes.keys()), min(args.num_classes, len(food_classes)))
    non_food_selected_classes = random.sample(list(non_food_classes.keys()), min(args.num_classes, len(non_food_classes)))

    # Preprocess and save images
    print("Preprocessing and saving food images...")
    preprocess_and_save_images(dataset, food_selected_classes, food_dir, args.images_per_class, food_classes)

    print("Preprocessing and saving non-food images...")
    preprocess_and_save_images(dataset, non_food_selected_classes, not_food_dir, args.images_per_class, non_food_classes)

    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and preprocess data for Food/Not Food classification")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Main data directory (relative to project root)")
    parser.add_argument("--food_subdir", type=str, default="food",
                        help="Subdirectory for food images (inside data directory)")
    parser.add_argument("--not_food_subdir", type=str, default="not_food",
                        help="Subdirectory for non-food images (inside data directory)")
    parser.add_argument("--num_classes", type=int, default=40,
                        help="Number of classes to include for each category")
    parser.add_argument("--images_per_class", type=int, default=100,
                        help="Number of images to include per class")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    args = parser.parse_args()

    # Set random seed for reproducibility
    random.seed(args.seed)

    main(args)
