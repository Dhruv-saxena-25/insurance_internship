import sys
import os 
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer


from sklearn.compose import ColumnTransformer

from insurance.constants import TARGET_COLUMN, SCHEMA_FILE_PATH 
from insurance.entity.config_entity import DataTransformationConfig
from insurance.entity.artifact_entity import DataTransformationArtifact, DataValidationArtifact, DataIngestionArtifact
from insurance.exception import InsuranceException
from insurance.logger import logging
from insurance.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file


class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
            self._schema_config= read_yaml_file(file_path= SCHEMA_FILE_PATH)
        except Exception as e:
            raise InsuranceException(e, sys) from e
        
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise InsuranceException(e, sys)
    
    def get_data_transformer_object(self) -> Pipeline:

        logging.info("Entered get_data_transformer_object method of DataTransformation class")

        try:
            logging.info("Got numerical cols from schema config")    
            
            numeric_transformer = StandardScaler()
            oh_transformer = OneHotEncoder()

            logging.info("Initialized StandardScaler, OneHotEncoder")

            oh_columns = self._schema_config['oh_columns']
            num_features = self._schema_config["num_features"]

            logging.info("Initialize PowerTransformer") ##  is a technique used to make numerical data resemble a Gaussian distribution more closely
            transform_pipe = Pipeline(steps=[('transformer', PowerTransformer(method='yeo-johnson'))])

            preprocessor = ColumnTransformer([
                    ("StandardScaler", numeric_transformer, num_features),
                    ("OneHotEncoder", oh_transformer, oh_columns)])
            
            logging.info("Created preprocessor object from ColumnTransformer")
            logging.info("Exited get_data_transformer_object method of DataTransformation class")
            return preprocessor
        except Exception as e:
            raise InsuranceException(e, sys) from e
        
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            if self.data_validation_artifact.validation_status:
                logging.info("Starting data transformation")
                preprocessor = self.get_data_transformer_object()
                logging.info("Got the preprocessor object")

                train_df = DataTransformation.read_data(file_path= self.data_ingestion_artifact.trained_file_path)
                test_df = DataTransformation.read_data(file_path= self.data_ingestion_artifact.test_file_path)

                ## Droping Target Column from training set.
                input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
                target_feature_train_final = train_df[TARGET_COLUMN]

                ## Droping Target Column from testing set.
                input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
                target_feature_test_final = test_df[TARGET_COLUMN]

                logging.info("Got train features and test features of Testing dataset")
                logging.info("Applying preprocessing object on training dataframe and testing dataframe")

                ## Applying Fit data
                input_feature_train_final = preprocessor.fit_transform(input_feature_train_df)
                logging.info("Used the preprocessor object to fit transform the train features")


                input_feature_test_final = preprocessor.transform(input_feature_test_df)
                logging.info("Used the preprocessor object to transform the test features")

                logging.info("Created train array and test array")

                train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final)]

                test_arr = np.c_[input_feature_test_final, np.array(target_feature_test_final)]

                save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
                save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
                save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)


                data_transformation_artifact = DataTransformationArtifact(
                    transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                    transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                    transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
                )
                return data_transformation_artifact
            else:
                raise Exception(self.data_validation_artifact.message)
        except Exception as e:
            raise InsuranceException(e, sys) from e
        

        