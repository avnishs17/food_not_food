# food_not_food

A deep learning project to classify images as food or not food.

---

## Overview

This project demonstrates a full machine learning workflow for binary image classification: distinguishing between food and non-food images. It features custom dataset curation, reproducible pipelines, and deployment-ready code.

---

## Dataset Curation & External Data Source

A key part of this project is the creation of a high-quality, custom dataset:

- **Curation Script:**  
  The script [`src/classifier/utils/download_and_preprocess.py`](src/classifier/utils/download_and_preprocess.py) downloads ImageNet data using the Hugging Face datasets library, filters it into food and non-food classes using WordNet, and saves the images into `data/food/` and `data/not_food/` directories.

- **How to Curate Locally:**  
  Run the following command to generate the dataset locally:
  ```bash
  python src/classifier/utils/download_and_preprocess.py
  ```
  This will create a `data/` directory in your project root with `food/` and `not_food/` subfolders containing the images.

- **External Data Source (Kaggle):**  
  To demonstrate reproducibility and the use of external data sources, the curated dataset is also uploaded to [Kaggle Datasets](https://www.kaggle.com/avnishs17/food-and-not-food) (see my Kaggle account).  
  **During the pipeline's data ingestion step, the dataset is automatically downloaded from Kaggle if not present locally.**  
  This ensures the workflow can be reproduced from a public data source, not just local files.

---

## Project Structure

```
food_not_food/
│
├── app.py, main.py                # Main application entry points
├── src/
│   └── classifier/
│       └── utils/
│           └── download_and_preprocess.py  # Dataset curation script
│       └── pipeline/              # Training and evaluation pipelines
│       └── ...                    # Other modules
├── artifacts/                     # Processed datasets, models, logs
├── data/                          # Curated food and not food images (local or from Kaggle)
├── requirements.txt               # Python dependencies
├── dvc.yaml, dvc.lock             # DVC pipeline files
├── research/                      # Jupyter notebooks for exploration
├── logs/                          # Training and running logs
└── ...
```

---

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/avnishs17/food_not_food.git
   cd food_not_food
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   uv venv
   .venv/Scripts/activate
   ```

3. **Install dependencies**
   ```bash
   uv pip install -r requirements.txt
   ```

4. **Curate the dataset (optional for demonstration)**
   ```bash
   python src/classifier/utils/download_and_preprocess.py
   ```
   This will create the `data/food/` and `data/not_food/` folders with images.

   > **Note:** For the full pipeline workflow, the data ingestion step will automatically download the dataset from Kaggle if it is not present locally.  
   > Make sure you have your Kaggle API credentials set up if running this step.

---

## Running the Project

### 1. Train the Model

```bash
python main.py
```
Or run the training pipeline scripts in `src/classifier/pipeline/` as needed.

### 2. Run the Web App

```bash
python app.py
```
Then open your browser at [http://localhost:8080](http://localhost:8080) (or the port specified in `app.py`).

### 3. DVC Commands

To initialize DVC, run the pipeline, and visualize the workflow:
```bash
dvc init
dvc repro
dvc dag
```

---

## Project Notebooks

- See the `research/` and `src/` folders for Jupyter notebooks demonstrating data ingestion, model training, and evaluation.

---

## AWS CI/CD Deployment with GitHub Actions

This project supports automated deployment using AWS EC2, ECR, and GitHub Actions. Below are the steps to set up continuous integration and deployment:

### 1. AWS Setup

- **Login to AWS Console.**
- **Create an IAM user** for deployment with the following permissions:
  - `AmazonEC2ContainerRegistryFullAccess`
  - `AmazonEC2FullAccess`
- **Create an ECR repository** to store your Docker image.
  - Save the URI: `XXXXXXXXXXXX.dkr.ecr.us-east-1.amazonaws.com/food_not_food`
- **Create an EC2 instance** (Ubuntu recommended).

### 2. Install Docker on EC2

```bash
# Update packages
sudo apt-get update -y
sudo apt-get upgrade

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker
```

### 3. Configure GitHub Secrets

Add the following secrets to your GitHub repository:

```
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION=us-east-1
AWS_ECR_LOGIN_URI=XXXXXXXXXXXX.dkr.ecr.us-east-1.amazonaws.com
ECR_REPOSITORY_NAME=food_not_food
```

### 4. Set Up EC2 as a Self-Hosted Runner

- Go to your GitHub repository: **Settings > Actions > Runners > New self-hosted runner**
- Choose your OS and follow the instructions to register the runner on your EC2 instance.

### 5. Push Dockerfile to GitHub

- Commit and push your Dockerfile and code to the repository.
- The GitHub Actions workflow will build and push the Docker image to ECR, and deploy/run it on your EC2 instance.

> You can monitor the deployment status on the GitHub Actions page.

---

## Reproducibility

- **Data:** Downloaded from Kaggle or curated locally.
- **Pipelines:** Managed with DVC for full reproducibility.
- **Environment:** All dependencies are listed in `requirements.txt`.

---

## Credits

- ImageNet, Hugging Face Datasets, WordNet, and Kaggle for data sources and APIs.
- Bootstrap and Flask for the web interface.

---

