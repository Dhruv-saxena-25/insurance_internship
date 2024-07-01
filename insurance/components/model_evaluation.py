import os, sys
import pickle
import numpy as np
import pandas as pd
from insurance.logger import logging
from insurance.exception import InsuranceException
from insurance.constants import *
from insurance.configuration.gcloud_syncer import GCloudSync
from sklearn.metrics import r2_score
from dataclasses import dataclass
from insurance.entity.config_entity import ModelEvaluationConfig
from insurance.utils.main_utils import load_numpy_array_data
from insurance.entity.artifact_entity import ModelEvaluationArtifact, ModelTrainerArtifact, DataTransformationArtifact, DataIngestionArtifact


class ModelEvaluation:
    def __init__(self, model_evaluation_config: ModelEvaluationConfig,
                 model_trainer_artifact: ModelTrainerArtifact,
                 #data_transformation_artifact: DataTransformationArtifact,
                 data_ingestion_artifact: DataIngestionArtifact):
                
                self.model_evaluation_config = model_evaluation_config
                self.model_trainer_artifact = model_trainer_artifact
                #self.data_transformation_artifact = data_transformation_artifact
                self.data_ingestion_artifact = data_ingestion_artifact
                self.gcloud = GCloudSync()

    def get_best_model_from_gcloud(self) -> str:

        try:
            logging.info("Entered the get_best_model_from_gcloud method of Model Evaluation class")
            os.makedirs(self.model_evaluation_config.best_model_dir, exist_ok=True)

            self.gcloud.sync_folder_from_gcloud(self.model_evaluation_config.bucket_name,
                                                self.model_evaluation_config.model_name,
                                                self.model_evaluation_config.best_model_dir)
            
            best_model_path = os.path.join(self.model_evaluation_config.best_model_dir,
                                           self.model_evaluation_config.model_name)

            logging.info("Exited the get_best_model_from_gcloud method of Model Evaluation class")
            return best_model_path
        except Exception as e:
            raise InsuranceException(e, sys) from e
        
    def get_trained_model(self)-> str:
         
        try:
            logging.info("Entered into file path to get trained model")

            trained_model = self.model_trainer_artifact.trained_model_file_path
            
            logging.info("Exited into file path to get trained model")
            
            return trained_model
        except Exception as e:
             raise InsuranceException(e, sys) from e


    def evaluate_model(self, model):

        try:
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            x_test, y_test = test_df.drop(TARGET_COLUMN, axis=1), test_df[TARGET_COLUMN]

            # x_test = load_numpy_array_data(file_path = self.data_transformation_artifact.transformed_test_file_path)
            
            with open(model, "rb") as file:
                 b_model = pickle.load(file)

            y_hat_model = b_model.predict(x_test)
            model_r2_score = r2_score(y_test, y_hat_model)
            
            return  model_r2_score
        except Exception as e:
            raise InsuranceException(e, sys) from e
        
    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        logging.info("Initiate Model Evaluation")
        try:
            
            logging.info("Loading currently trained model")
            trained_model = self.get_trained_model()
            trained_model_r2_score = self.evaluate_model(trained_model)

            logging.info(f"Fetch best model from gcloud storage {trained_model_r2_score}")
            best_model_path = self.get_best_model_from_gcloud()
            
            logging.info("Check is best model present in the gcloud storage or not ?")
            if os.path.isfile(best_model_path) is False:
                is_model_accepted = True
                logging.info("glcoud storage model is false and currently trained model accepted is true")
            
            else:
                logging.info("Load best model fetched from gcloud storage")
                gcloud_model_r2_score = self.evaluate_model(best_model_path)
                logging.info(f"Comparing loss between best_model_loss and trained_model_loss ? {gcloud_model_r2_score} ")

                if gcloud_model_r2_score > trained_model_r2_score:
                    is_model_accepted = True
                    logging.info("Trained model not accepted")
                else:
                    is_model_accepted = False
                    logging.info("Trained model accepted")

            model_evaluation_artifacts = ModelEvaluationArtifact(is_model_accepted=is_model_accepted)
            logging.info("Returning the ModelEvaluationArtifacts")
            return model_evaluation_artifacts



            
            
        except Exception as e:
            raise InsuranceException(e, sys) from e
        