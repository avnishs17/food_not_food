# Food/Not Food Data Downloader

This script downloads and preprocesses data for the Food/Not Food classification project.

## Overview

The script:
1. Downloads ImageNet data using the Hugging Face datasets library
2. Filters the data into food and non-food classes using WordNet
3. Saves the images to appropriate directories

## Requirements

- Python 3.6+
- huggingface-hub
- datasets
- nltk
- Pillow (PIL)
- tqdm

You can install the required packages with:

```bash
pip install huggingface-hub datasets nltk Pillow tqdm
```

## Usage

Basic usage:

```bash
python download_and_preprocess.py
```

This will:
- Download ImageNet data
- Filter it into food and non-food classes
- Save 100 images per class for 40 classes in each category
- Create the directory structure:
  - `food_not_food_data/food/` - Food images
  - `food_not_food_data/not_food/` - Non-food images
  - `food_not_food_data/food_classes.json` - Mapping of food class IDs to names
  - `food_not_food_data/non_food_classes.json` - Mapping of non-food class IDs to names

### Command-line Arguments

You can customize the behavior with these arguments:

- `--output_dir`: Directory to save data to (default: "food_not_food_data")
- `--num_classes`: Number of classes to include for each category (default: 40)
- `--images_per_class`: Number of images to include per class (default: 100)
- `--seed`: Random seed for reproducibility (default: 42)

Example with custom settings:

```bash
python download_and_preprocess.py --output_dir custom_data --num_classes 20 --images_per_class 50
```

## Notes

- The script uses WordNet to identify food-related classes in ImageNet
- Some manual corrections are applied to ensure proper classification
- The download process may take a while depending on your internet connection
- The script requires approximately 800MB of disk space for the default settings (40 classes × 100 images × 2 categories)

## Troubleshooting

If you encounter issues with the Hugging Face datasets library, make sure you have:
1. A Hugging Face account
2. Accepted the ImageNet terms of use on the Hugging Face website
3. Logged in with `huggingface-cli login`
