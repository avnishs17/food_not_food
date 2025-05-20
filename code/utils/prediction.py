"""
Prediction and evaluation utilities for the Food/Not Food classification project.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Add the parent directory to the path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def preprocess_image(image_path, target_size=None):
    """
    Preprocess an image for prediction.
    
    Args:
        image_path (str): Path to the image file
        target_size (tuple, optional): Target size for the image. Defaults to config.IMAGE_SIZE.
    
    Returns:
        numpy.ndarray: Preprocessed image array
    """
    if target_size is None:
        target_size = config.IMAGE_SIZE
    
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Create a batch
    img_array /= 255.0  # Normalize to [0,1]
    
    return img_array, img


def predict_image(model, image_path, target_size=None):
    """
    Make a prediction for a single image.
    
    Args:
        model (tf.keras.Model): Model to use for prediction
        image_path (str): Path to the image file
        target_size (tuple, optional): Target size for the image. Defaults to config.IMAGE_SIZE.
    
    Returns:
        tuple: (predicted_class_index, predicted_class_name, confidence)
    """
    # Preprocess the image
    img_array, img = preprocess_image(image_path, target_size)
    
    # Make a prediction
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence = predictions[0][predicted_class_index]
    
    # Get the class name
    predicted_class_name = config.CLASS_MAPPING[predicted_class_index]
    
    return predicted_class_index, predicted_class_name, confidence, img


def display_prediction(image_path, model, target_size=None):
    """
    Display an image with its prediction.
    
    Args:
        image_path (str): Path to the image file
        model (tf.keras.Model): Model to use for prediction
        target_size (tuple, optional): Target size for the image. Defaults to config.IMAGE_SIZE.
    """
    # Make a prediction
    predicted_class_index, predicted_class_name, confidence, img = predict_image(model, image_path, target_size)
    
    # Display the image with the prediction
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.title(f'Predicted: {predicted_class_name} ({confidence:.2f})')
    plt.axis('off')  # Hide axes
    plt.show()


def evaluate_model(model, test_dataset):
    """
    Evaluate the model on a test dataset.
    
    Args:
        model (tf.keras.Model): Model to evaluate
        test_dataset (tf.data.Dataset): Test dataset
    
    Returns:
        tuple: (loss, accuracy)
    """
    # Evaluate the model
    loss, accuracy = model.evaluate(test_dataset)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    return loss, accuracy


def predict_batch(model, image_paths, target_size=None):
    """
    Make predictions for a batch of images.
    
    Args:
        model (tf.keras.Model): Model to use for prediction
        image_paths (list): List of paths to image files
        target_size (tuple, optional): Target size for the images. Defaults to config.IMAGE_SIZE.
    
    Returns:
        list: List of (image_path, predicted_class_name, confidence) tuples
    """
    results = []
    
    for image_path in image_paths:
        # Make a prediction
        predicted_class_index, predicted_class_name, confidence, _ = predict_image(model, image_path, target_size)
        
        # Add to results
        results.append((image_path, predicted_class_name, confidence))
    
    return results


def plot_training_history(history):
    """
    Plot the training history.
    
    Args:
        history (tf.keras.callbacks.History): Training history
    """
    # Plot accuracy
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    from models.model import load_model
    
    # Load model
    model = load_model()
    
    # Example image path
    image_path = "path/to/your/image.jpg"
    
    # Display prediction
    display_prediction(image_path, model)
