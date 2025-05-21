# food_not_food

A deep learning project to classify images as food or not food.

## Project Structure

- `app.py` / `main.py`: Main application entry points.
- `artifacts/`: Contains datasets, models, and logs.
- `code/`, `src/`: Source code for data processing, model training, and utilities.
- `requirements.txt`: Python dependencies.
- `dvc.yaml`, `dvc.lock`: DVC pipeline files for data and model versioning.

## Setup Instructions

1. **Clone the repository**

   ```powershell
   git clone https://github.com/avnishs17/food_not_food.git
   cd food_not_food
   ```
2. **Create a virtual environment (optional but recommended)**

   ```powershell
    uv venv    # you need have uv package in your base environment.
   .venv/Scripts/activate
   ```
3. **Install dependencies**

   ```powershell
   uv pip install -r requirements.txt
   ```

## Running the Project

### 1. Train the Model

```powershell
python main.py
```

Or run the training pipeline scripts in `src/classifier/pipeline/` as needed.

### 2. Run the Web App (if available)

```powershell
python app.py
```

Then open your browser at [http://localhost:80](http://localhost:80) (or the port specified in `app.py`).

### 3. DVC commands

To initialize dvc after creating dvc.yaml and run the whole pipeline through dvc repro. dvc dag shows the command in Data Version Control (DVC) is used to visualize the pipeline(s).

```bash
    dvc init
    dvc repro
    dvc dag
```

## Project Notebooks

- See the `research/` and `code/` folders for Jupyter notebooks demonstrating data ingestion, model training, and evaluation.

## Project Organization

- `src/classifier/`: Modular code for data ingestion, model preparation, training, evaluation, and prediction.
- `artifacts/`: Stores datasets, models, and logs.
- `logs/`: Training and running logs.

## AWS CICD Deployment with GitHub Actions

### Setup Steps

1. Login to AWS console.
2. Create IAM user for deployment with specific access:

   - EC2 access: Virtual machine
   - ECR: Elastic Container Registry to save your docker image in AWS
3. Add the following policies to the IAM user:

   - `AmazonEC2ContainerRegistryFullAccess`
   - `AmazonEC2FullAccess`
4. Create ECR repository to store/save docker image

   - Save the URI: `XXXXXXXXXXXX.dkr.ecr.us-east-1.amazonaws.com/food_not_food`
5. Create EC2 machine (Ubuntu)
6. Open EC2 and Install docker in EC2 Machine:

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
7. Setup GitHub secrets:

   ```
   AWS_ACCESS_KEY_ID=
   AWS_SECRET_ACCESS_KEY=
   AWS_REGION = us-east-1
   AWS_ECR_LOGIN_URI = XXXXXXXXXXXX.dkr.ecr.us-east-1.amazonaws.com
   ECR_REPOSITORY_NAME = food_not_food
   ```
8. Configure EC2 as self-hosted runner:

   - Go to GitHub repository
   - Settings > Actions > Runners > New self-hosted runner
   - Choose OS
   - Run the commands provided by GitHub one by one


9. Push Dockerfile to github repo
``` 
- ./run.sh command given runner setting should running in aws cli instance
- You can check github repo actions page for status.
```

