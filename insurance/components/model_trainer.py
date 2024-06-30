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
from insurance.entity.estimator import InsuranceModel



class ModelTrainer:

    def __init__(self,data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config


    def get_model_object_and_report(self, train: np.array, test: np.array) -> Tuple[object, object]:

        try:
            logging.info("Using neuro_mf to get best model object and report")
            model_factory = ModelFactory(model_config_path= self.model_trainer_config.model_config_file_path)

            x_train, y_train, x_test, y_test = train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]

            best_model_detail = model_factory.get_best_model(X=x_train,y=y_train,
                                                             base_accuracy=self.model_trainer_config.expected_r2_score)
            
            model_obj = best_model_detail.best_model
            y_pred = model_obj.predict(x_test)

            mae= mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            metric_artifact = RegressionMetricArtifact(mae = mae,
                                                       mse= mse,
                                                       r2_score= r2)
            return best_model_detail, metric_artifact 
        except Exception as e:
            raise InsuranceException(e, sys) from e
        

    def initiate_model_trainer(self):
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")
        try:
            train_arr = load_numpy_array_data(file_path= self.data_transformation_artifact.transformed_train_file_path)
            test_arr =  load_numpy_array_data(file_path = self.data_transformation_artifact.transformed_test_file_path)

            best_model_detail, metric_artifact = self.get_model_object_and_report(train=train_arr, test=test_arr)
            preprocessing_obj = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)


            if best_model_detail.best_score < self.model_trainer_config.expected_r2_score:
                logging.info("No best model found with score more than base score")
                raise Exception("No best model found with score more than base score")
            
            insurance_model = InsuranceModel(preprocessing_object=preprocessing_obj,
                                       trained_model_object=best_model_detail.best_model)
            
            logging.info("Created usvisa model object with preprocessor and model")
            logging.info("Created best model file path.")

            save_object(self.model_trainer_config.trained_model_file_path, insurance_model)

            model_trainer_artifact = ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                                                          metric_artifact=metric_artifact)
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise InsuranceException(e, sys) from e