"""
Main script for the Food/Not Food classification project.
"""

import os
import argparse
import tensorflow as tf
import wandb
from preprocessing.data_preparation import create_directory_structure, copy_images, create_tf_datasets
from models.model import enable_mixed_precision, setup_gpu, create_model, train_model, save_model, load_model, convert_to_tflite
from utils.prediction import evaluate_model, display_prediction, plot_training_history
import config


def setup_wandb(project_name="food_not_food"):
    """
    Setup Weights & Biases for experiment tracking.
    
    Args:
        project_name (str, optional): Name of the project. Defaults to "food_not_food".
    
    Returns:
        wandb.config: WandB configuration
    """
    # Initialize WandB
    wandb.init(
        project=project_name, 
        config={
            "learning_rate": config.LEARNING_RATE, 
            "architecture": config.MODEL_ARCHITECTURE,
            "dataset": "Food-101 and filtered ImageNet-1k",
            "epochs": config.EPOCHS
        }
    )
    
    return wandb.config


def create_wandb_callback():
    """
    Create a WandB callback for logging metrics during training.
    
    Returns:
        wandb.keras.WandbCallback: WandB callback
    """
    return wandb.keras.WandbCallback(
        monitor="val_accuracy",
        log_weights=True,
        log_evaluation=True,
        save_model=False
    )


def prepare_data(food_dirs, not_food_dir):
    """
    Prepare the data for training and testing.
    
    Args:
        food_dirs (list): List of directories containing food images
        not_food_dir (str): Directory containing non-food images
    """
    # Create directory structure
    create_directory_structure()
    
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


def train(use_wandb=False):
    """
    Train the model.
    
    Args:
        use_wandb (bool, optional): Whether to use Weights & Biases for tracking. Defaults to False.
    
    Returns:
        tuple: (model, history)
    """
    # Setup
    enable_mixed_precision()
    setup_gpu()
    
    # Create datasets
    train_dataset, val_dataset = create_tf_datasets()
    
    # Create model
    model = create_model()
    
    # Setup WandB if requested
    callbacks = []
    if use_wandb:
        setup_wandb()
        callbacks.append(create_wandb_callback())
    
    # Train model
    history = train_model(model, train_dataset, val_dataset, callbacks=callbacks)
    
    # Save model
    save_model(model)
    
    # Convert to TFLite
    convert_to_tflite(model)
    
    return model, history


def evaluate(model_path=None):
    """
    Evaluate the model.
    
    Args:
        model_path (str, optional): Path to the model to evaluate. Defaults to config.MODEL_SAVE_PATH.
    
    Returns:
        tuple: (loss, accuracy)
    """
    # Load model
    model = load_model(model_path)
    
    # Create datasets
    _, val_dataset = create_tf_datasets()
    
    # Evaluate model
    return evaluate_model(model, val_dataset)


def predict(image_path, model_path=None):
    """
    Make a prediction for a single image.
    
    Args:
        image_path (str): Path to the image file
        model_path (str, optional): Path to the model to use. Defaults to config.MODEL_SAVE_PATH.
    """
    # Load model
    model = load_model(model_path)
    
    # Display prediction
    display_prediction(image_path, model)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Food/Not Food Classification')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Prepare data command
    prepare_parser = subparsers.add_parser('prepare', help='Prepare data for training')
    prepare_parser.add_argument('--food-dirs', nargs='+', default=['food-101/images', 'imagenet_food_images'], 
                               help='Directories containing food images')
    prepare_parser.add_argument('--not-food-dir', default='imagenet_not_food_images', 
                               help='Directory containing non-food images')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--wandb', action='store_true', help='Use Weights & Biases for tracking')
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate the model')
    evaluate_parser.add_argument('--model-path', default=None, help='Path to the model to evaluate')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make a prediction for a single image')
    predict_parser.add_argument('image_path', help='Path to the image file')
    predict_parser.add_argument('--model-path', default=None, help='Path to the model to use')
    
    args = parser.parse_args()
    
    if args.command == 'prepare':
        prepare_data(args.food_dirs, args.not_food_dir)
    elif args.command == 'train':
        model, history = train(args.wandb)
        plot_training_history(history)
    elif args.command == 'evaluate':
        evaluate(args.model_path)
    elif args.command == 'predict':
        predict(args.image_path, args.model_path)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
