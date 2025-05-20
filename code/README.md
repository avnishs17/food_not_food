# Food/Not Food Classification

This is a modular implementation of a food/not food image classification system using TensorFlow and EfficientNetV2.

## Project Structure

```
code/
├── config.py                  # Configuration settings
├── main.py                    # Main script
├── data/                      # Data directory
├── models/                    # Model definitions
│   ├── __init__.py
│   └── model.py               # Model creation and training
├── preprocessing/             # Data preprocessing
│   ├── __init__.py
│   └── data_preparation.py    # Data preparation utilities
└── utils/                     # Utility functions
    ├── __init__.py
    └── prediction.py          # Prediction and evaluation utilities
```

## Requirements

- Python 3.8+
- TensorFlow 2.x
- NumPy
- Matplotlib
- NLTK
- Weights & Biases (optional, for experiment tracking)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/food_not_food.git
cd food_not_food
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Data Preparation

To prepare the data for training:

```bash
python -m code.main prepare --food-dirs path/to/food/images --not-food-dir path/to/not_food/images
```

### Training

To train the model:

```bash
python -m code.main train
```

To train with Weights & Biases tracking:

```bash
python -m code.main train --wandb
```

### Evaluation

To evaluate the model:

```bash
python -m code.main evaluate
```

To evaluate a specific model:

```bash
python -m code.main evaluate --model-path path/to/model.h5
```

### Prediction

To make a prediction for a single image:

```bash
python -m code.main predict path/to/image.jpg
```

To use a specific model for prediction:

```bash
python -m code.main predict path/to/image.jpg --model-path path/to/model.h5
```

## Configuration

You can modify the configuration settings in `config.py` to adjust:

- Image size
- Batch size
- Number of epochs
- Learning rate
- Data augmentation settings
- Model architecture
- And more

## Model

The model uses EfficientNetV2B0 as the base model with a global average pooling layer and a dense output layer with softmax activation.

## Data Augmentation

The training data is augmented with:

- Random horizontal flips
- Random rotations
- Random zooms

## License

This project is licensed under the MIT License - see the LICENSE file for details.
