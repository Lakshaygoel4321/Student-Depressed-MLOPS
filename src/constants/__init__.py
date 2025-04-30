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

TARGET_COLUMN = "Depression"
PROCESSING_OBJ_FILE = "preprocessor.pkl"
SCHEMA_FILE_PATH = os.path.join("config","schema.yaml")

# data ingestion
DATA_INGESTION_COLLECTION_NAME: str = "Proj1-Data"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2

# data validation
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"

# data transformation
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRAMSFORMED_OBJECT_DIR: str = "transformed_object"

# model trainer
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6
MODEL_TRAINER_MODEL_CONFIG_FILE_PATH: str = os.path.join('config','model.yaml')


