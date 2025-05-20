import os
import zipfile
from dotenv import load_dotenv
from pathlib import Path
from classifier import logger
from classifier.utils.common import get_size
from classifier.entity.config_entity import DataIngestionConfig

load_dotenv()
from kaggle.api.kaggle_api_extended import KaggleApi

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            api = KaggleApi()
            api.authenticate()

            logger.info(f"Downloading dataset: {self.config.dataset_name}")
            api.dataset_download_files(
                dataset=self.config.dataset_name,
                path=self.config.root_dir,
                # filename=os.path.basename(self.config.local_data_file),
                unzip=False
            )
            downloaded_zip_path = os.path.join(self.config.root_dir, f"{self.config.dataset_name.split('/')[-1]}.zip")
            os.rename(downloaded_zip_path, self.config.local_data_file)
            logger.info(f"Downloaded and renamed to: {self.config.local_data_file}")
        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")

    def extract_zip_file(self):
        os.makedirs(self.config.unzip_dir, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(self.config.unzip_dir)
            logger.info(f"Extracted to {self.config.unzip_dir}")
