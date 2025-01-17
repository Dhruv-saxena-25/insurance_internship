import sys
from typing import Tuple
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from neuro_mf import ModelFactory
from insurance.exception import InsuranceException
from insurance.logger import logging
from insurance.utils.main_utils import load_numpy_array_data, load_object, save_object, read_yaml_file
from insurance.entity.config_entity import ModelTrainerConfig
from insurance.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, RegressionMetricArtifact
from insurance.entity.estimator import InsuranceModel, get_regression_metric
from catboost import CatBoostRegressor

class ModelTrainer:
    def __init__(self,data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def train_model(self, x_train, y_train):
        try:

            model = CatBoostRegressor(depth = 3, iterations= 100, learning_rate= 0.1, verbose=False)
            model.fit(x_train, y_train)

            
            return model
        except Exception as e:
            raise InsuranceException(e, sys) from e
        
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_arr = load_numpy_array_data(file_path= self.data_transformation_artifact.transformed_train_file_path)
            test_arr =  load_numpy_array_data(file_path = self.data_transformation_artifact.transformed_test_file_path)


            x_train, y_train, x_test, y_test= (

                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )
            model= self.train_model(x_train, y_train)
            y_train_pred= model.predict(x_train)
            
            regression_metric = get_regression_metric(y_train, y_train_pred)
            
           
            # reg_train_metric = regression_metric.r2_score(y_train, y_train_pred)

            if regression_metric.r2_score<=self.model_trainer_config.expected_r2_score:
                logging.info("No best model found with score more than base score")
                raise Exception("No best model found with score more than base score")
            
            # y_test_pred= model.predict(x_test)
            # reg_test_metric = metric_artifact.r2_score(y_test, y_test_pred)

            preprocessor= load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)

            

            insurance_model = InsuranceModel(preprocessing_object=preprocessor,
                                       trained_model_object=model)

            



            logging.info("Created Insurance model object with preprocessor and model")
            logging.info("Created best model file path.")

            save_object(self.model_trainer_config.trained_model_file_path, insurance_model)

            model_trainer_artifact = ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                                                          metric_artifact=regression_metric)
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise InsuranceException(e, sys) from e