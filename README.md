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
Then open your browser at [http://localhost:5000](http://localhost:5000) (or the port specified in `app.py`).

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
