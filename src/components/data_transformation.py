import os
import numpy as np
import sys
import pandas as pd
import numpy as np

from src.logger import logging
from src.exception import USvisaException
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact,DataIngestionArtifact,DataValidationArtifact
from src.constants import *
from sklearn.preprocessing import StandardScaler,OneHotEncoder,OrdinalEncoder
from src.utils.main_utils import *
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                data_transformation_config: DataTransformationConfig,
                data_validation_artifact: DataValidationArtifact
                ):
        
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self.schema_path = read_yaml_file(file_path=SCHEMA_FILE_PATH)

        except Exception as e:
            raise USvisaException(sys,e)
        
    
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise USvisaException(e, sys)
        
    @staticmethod
    def clean_category(cat):
        return cat.strip("'")
        

    def get_data_transformed(self) -> Pipeline:
        logging.info("entered the data transformation")

        try:
            logging.info("Got numerical col from schema config")

            scaler = StandardScaler()
            encoder = OneHotEncoder(sparse_output=False)
            ordinal_encoder = OrdinalEncoder()

            onehot = self.schema_path['onehotEncoder']

            ordinal = self.schema_path['ordinalEncoder']

            num_columns = self.schema_path['scaler_columns']
            
            logging.info("Intializing the standardscaler, onehotencoder")

            preprocessor = ColumnTransformer([
                ('onehotencoder',encoder,onehot),
                ('Ordinalencoder',ordinal_encoder,ordinal),
                ("StandardScaler",scaler,num_columns)

            ])

            logging.info("Creating preprocessing object from columntransformer")

            logging.info(
                "exited get_data_transformed method of data_transformation class"

            )

            return preprocessor
        
        except Exception as e:
            raise USvisaException(e, sys) from e


    

    def initiate_data_transformation(self,) -> DataTransformationArtifact:
        try:
            
            if self.data_validation_artifact.validation_status:
                logging.info("Starting data transformation")
                preprocessor = self.get_data_transformed()
                logging.info("Got the preprocessor object")

                train_df = DataTransformation.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
                test_df = DataTransformation.read_data(file_path=self.data_ingestion_artifact.test_file_path)

                input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN],axis=1)
                target_train_df = train_df[TARGET_COLUMN]

                logging.info("Got the training feature and testing feature in the dataset")

                drop_col = self.schema_path['drop_columns']

                logging.info("Droping the column in the training dataset")
                input_feature_train_df = drop_columns(df=input_feature_train_df,cols=drop_col)
                
                logging.info("Converting the age datatype into int datatype")
                input_feature_train_df['Age'] = input_feature_train_df['Age'].astype('int')

                logging.info("converting in profession feature multiple string into the single string")
                profession_counts = input_feature_train_df['Profession'].value_counts()
                common_professions = profession_counts[profession_counts > 500].index

                input_feature_train_df['Profession'] = input_feature_train_df['Profession'].apply(
                    lambda x: x if x in common_professions else 'Other_Professions'
                )
                logging.info("converting in Sleep Duration multiple string into the single string")
                input_feature_train_df['Sleep Duration'] = input_feature_train_df['Sleep Duration'].apply(self.clean_category)

                logging.info("Replace the ? to the 1.0 in the financial stress")
                input_feature_train_df['Financial Stress'] = input_feature_train_df['Financial Stress'].replace("?","1.0")
        
                logging.info("Now coming on the degree feature")
                logging.info("selecting a threshold then extract the degree of value count threshold is 500")
                
                degree = input_feature_train_df['Degree'].value_counts()
                degree = degree[degree>500]
                
                def degree_function(text):
                    if text in degree:
                        return text
                    else:
                        return "Others_Degree"
                    
                input_feature_train_df['Degree'] = input_feature_train_df['Degree'].apply(degree_function)

                logging.info("Now coming on the city feature")
                city = input_feature_train_df['City'].value_counts()
                city = city[city>500]

                def city_function(text):
                    if text in city:
                        return text
                    else:
                        return "Other_Citys"
                    
                input_feature_train_df['City'] = input_feature_train_df['City'].apply(city_function)


                logging.info("Now working on the testing data")

                input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN],axis=1)
                target_test_df = test_df[TARGET_COLUMN]

                logging.info("Droping the column into the testing data")
                input_feature_test_df = drop_columns(df=input_feature_test_df,cols=drop_col)
                
                logging.info("Converting the age datatype into int datatype")
                input_feature_test_df['Age'] = input_feature_test_df['Age'].astype('int')

                logging.info("converting in profession feature multiple string into the single string in testing")
                # input_feature_test_df['Profession'] = input_feature_test_df['Profession'].apply(self.clean_category)
                input_feature_test_df['Profession'] = input_feature_test_df['Profession'].apply(
                lambda x: x if x in common_professions else 'Other_Professions'
                )

                logging.info("converting in Sleep Duration multiple string into the single string")
                input_feature_test_df['Sleep Duration'] = input_feature_test_df['Sleep Duration'].apply(self.clean_category)

                logging.info("Replace the ? to the 1.0 in the financial stress")
                input_feature_test_df['Financial Stress'] = input_feature_test_df['Financial Stress'].replace("?","1.0")
        
                logging.info("Now coming on the degree feature")
                logging.info("selecting a threshold then extract the degree of value count threshold is 500")
                
                input_feature_test_df['Degree'] = input_feature_test_df['Degree'].apply(degree_function)

                logging.info("Now coming on the city feature")
                input_feature_test_df['City'] = input_feature_test_df['City'].apply(city_function)

                print(input_feature_train_df.columns)
                print(input_feature_test_df.columns)

                logging.info("Now applying the preprocessor object")
                input_feature_train = preprocessor.fit_transform(input_feature_train_df)
                
                logging.info("Applying on the preprocessor object on the testing")
                input_feature_test = preprocessor.transform(input_feature_test_df)

                logging.info("Created train array and test array")
                train_array = np.c_[
                    input_feature_train , np.array(target_train_df)
                ]

                test_array = np.c_[
                    input_feature_test , np.array(target_test_df)
                ]

                save_object(self.data_transformation_config.transformed_object_file_path,preprocessor)
                save_numpy_array_data(self.data_transformation_config.transformed_train_file_path,train_array)
                save_numpy_array_data(self.data_transformation_config.transformed_test_file_path,test_array)

                logging.info("Saved the preprocessing")
                logging.info("Exited initiate data_transformation method of data_transformation")

                data_transformation_artifact = DataTransformationArtifact(
                    transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                    transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                    transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
                )
                return data_transformation_artifact
            else:
                raise Exception(self.data_validation_artifact.message)


        except Exception as e:
            raise USvisaException(e, sys) from e