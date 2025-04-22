import os
from datetime import date

DATABASE_NAME = 'Proj1'

COLLECTION_NAME = 'Proj1-Data'

MONGODB_URL_KEY = "MONGODB_URL"

PIPELINE_NAME: str = 'student'
ARTIFACT_DIR: str = 'artifact'

TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

FILE_NAME:str = 'student.csv'
MODEL_FILE_NAME = 'model.pkl'

TARGET_COLUMN = ""
PROCESSING_OBJ_FILE = "preprocessor.pkl"
SCHEMA_FILE_PATH = os.path.join("config","schema.yaml")

DATA_INGESTION_COLLECTION_NAME: str = "Proj1-Data"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2




