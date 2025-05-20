"""
Model definition and training utilities for the Food/Not Food classification project.
"""

import os
import sys
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras import mixed_precision

# Add the parent directory to the path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def enable_mixed_precision():
    """Enable mixed precision training for faster training."""
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    print("Mixed precision enabled")


def setup_gpu():
    """Setup GPU for training if available."""
    gpus = tf.config.list_physical_devices('GPU')
    print(f"Num GPUs Available: {len(gpus)}")
    
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def create_model():
    """
    Create and compile the model for food/not-food classification.
    
    Returns:
        tf.keras.Model: Compiled model
    """
    # Define the model
    base_model = EfficientNetV2B0(
        include_top=False, 
        input_shape=(*config.IMAGE_SIZE, 3), 
        weights=config.MODEL_WEIGHTS
    )
    
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    output = tf.keras.layers.Dense(config.NUM_CLASSES, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=base_model.input, outputs=output)
    
    # Compile the model
    model.compile(
        optimizer='adam', 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    return model


def train_model(model, train_dataset, val_dataset, epochs=None, callbacks=None):
    """
    Train the model on the provided datasets.
    
    Args:
        model (tf.keras.Model): Model to train
        train_dataset (tf.data.Dataset): Training dataset
        val_dataset (tf.data.Dataset): Validation dataset
        epochs (int, optional): Number of epochs to train for. Defaults to config.EPOCHS.
        callbacks (list, optional): List of callbacks to use during training. Defaults to None.
    
    Returns:
        tf.keras.callbacks.History: Training history
    """
    if epochs is None:
        epochs = config.EPOCHS
    
    # Train the model
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        callbacks=callbacks
    )
    
    return history


def save_model(model, filepath=None):
    """
    Save the model to disk.
    
    Args:
        model (tf.keras.Model): Model to save
        filepath (str, optional): Path to save the model to. Defaults to config.MODEL_SAVE_PATH.
    """
    if filepath is None:
        filepath = config.MODEL_SAVE_PATH
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save the model
    model.save(filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath=None):
    """
    Load a model from disk.
    
    Args:
        filepath (str, optional): Path to load the model from. Defaults to config.MODEL_SAVE_PATH.
    
    Returns:
        tf.keras.Model: Loaded model
    """
    if filepath is None:
        filepath = config.MODEL_SAVE_PATH
    
    # Load the model
    model = tf.keras.models.load_model(filepath)
    print(f"Model loaded from {filepath}")
    
    return model


def convert_to_tflite(model, filepath=None):
    """
    Convert the model to TFLite format.
    
    Args:
        model (tf.keras.Model): Model to convert
        filepath (str, optional): Path to save the TFLite model to. Defaults to config.TFLITE_MODEL_PATH.
    """
    if filepath is None:
        filepath = config.TFLITE_MODEL_PATH
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Convert the model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    # Save the model
    with open(filepath, 'wb') as f:
        f.write(tflite_model)
    
    print(f"TFLite model saved to {filepath}")


if __name__ == "__main__":
    # Example usage
    from preprocessing.data_preparation import create_tf_datasets
    
    # Setup
    enable_mixed_precision()
    setup_gpu()
    
    # Create datasets
    train_dataset, val_dataset = create_tf_datasets()
    
    # Create and train model
    model = create_model()
    history = train_model(model, train_dataset, val_dataset)
    
    # Save model
    save_model(model)
    
    # Convert to TFLite
    convert_to_tflite(model)
