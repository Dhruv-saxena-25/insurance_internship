import os
import sys
import pickle 
from insurance.constants import *
from insurance.logger import logging
from insurance.exception import InsuranceException
from insurance.configuration.gcloud_syncer import GCloudSync
from insurance.components.data_transformation import DataTransformation
from insurance.entity.config_entity import DataTransformationConfig
from insurance.entity.artifact_entity import DataIngestionArtifact
from insurance.entity.config_entity import TrainingPipelineConfig
from insurance.utils.main_utils import get_preprocessor_path, load_object

training_pipeline_config: TrainingPipelineConfig= TrainingPipelineConfig()

class PredictionPipeline:
    def __init__(self):
        self.bucket_name = BUCKET_NAME
        self.model_name = MODEL_NAME
        self.model_path = os.path.join(training_pipeline_config.artifact_dir, "PredictedModel")
        self.gcloud = GCloudSync()
        self.data_transformation = DataTransformation(data_transformation_config= DataTransformationConfig,
                                                      data_ingestion_artifacts=DataIngestionArtifact)

    def get_model_from_gcloud(self) -> str:
        
        logging.info("Entered the get_model_from_gcloud method of PredictionPipeline class")

        try:
             # Loading the best model from GCP bucket
             os.makedirs(self.model_path, exist_ok=True)
             self.gcloud.sync_folder_from_gcloud(self.bucket_name, self.model_name, self.model_path)
             best_model_path = os.path.join(self.model_path, self.model_name)
             logging.info("Exited the get_model_from_gcloud method of PredictionPipeline class")
             return best_model_path
        except Exception as e:
            raise InsuranceException(e, sys) from e
        
    def predict(self,features):
        try:
            best_model_path = self.get_model_from_gcloud()
            preprocessor_path = get_preprocessor_path()
            print(preprocessor_path)
            best_model = load_object(file_path= best_model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            
            print("After Loading")
            data_scaled= preprocessor.transform(features)
            preds= best_model.predict(data_scaled)
            return preds

        except Exception as e:
            raise InsuranceException(e, sys) from e 
        

    def run_pipeline(self, features):
        logging.info("Entered the run_pipeline method of PredictionPipeline class")
        try:
            pred_result = self.predict(features)
            logging.info("Exited the run_pipeline method of PredictionPipeline class")
            return pred_result
        except Exception as e:
            raise InsuranceException(e, sys) from e 
        