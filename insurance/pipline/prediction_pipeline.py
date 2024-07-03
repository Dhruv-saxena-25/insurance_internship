import os
import sys
import pickle 

from insurance.constants import *
from insurance.logger import logging
from insurance.exception import InsuranceException
from insurance.configuration.gcloud_syncer import GCloudSync
from insurance.entity.estimator import InsuranceModel
from insurance.utils.main_utils import get_preprocessor_path, load_object
import pandas as pd


class PredictionPipeline:
    def __init__(self):
        self.bucket_name = BUCKET_NAME
        self.model_name = MODEL_NAME
        self.model_path = os.path.join("PredictedModel")
        self.gcloud = GCloudSync()

    def get_model_from_gcloud(self) -> str:
        
        logging.info("Entered the get_model_from_gcloud method of PredictionPipeline class")

        try:
             # Loading the best model from GCP bucket
            os.makedirs(self.model_path, exist_ok=True)             
            best_model_path = os.path.join(self.model_path, self.model_name)

            if os.path.isfile(best_model_path) is False:
                logging.info("Downloading Model from GCP bucket!!!")
                self.gcloud.sync_folder_from_gcloud(self.bucket_name, self.model_name, self.model_path)

            logging.info("Exited the get_model_from_gcloud method of PredictionPipeline class")
            return best_model_path
        except Exception as e:
            raise InsuranceException(e, sys) from e
        
    def predicts(self,dataframe) -> float:
        logging.info("Entered the predict method of PredictionPipeline class")
        try:
            
            # if self.model_path is not False
            best_model_path = self.get_model_from_gcloud()

            best_model = load_object(file_path= best_model_path)
            
            print("After Loading Best Model")
            
            expenses= best_model.predict(dataframe)
            
            logging.info("Exited the predict method of PredictionPipeline class")
            return  expenses  
            
            
        except Exception as e:
            raise InsuranceException(e, sys) from e 
        
