import sys
from insurance.logger import logging
from insurance.exception import InsuranceException
from insurance.configuration.gcloud_syncer import GCloudSync
from insurance.entity.config_entity import ModelPusherConfig
from insurance.entity.artifact_entity import ModelPusherArtifacts


class ModelPusher:
    def __init__(self, model_pusher_config: ModelPusherConfig):
        
        self.model_pusher_config = model_pusher_config
        self.gcloud = GCloudSync()

    def initiate_model_pusher(self) -> ModelPusherArtifacts:

        logging.info("Entered initiate_model_pusher method of ModelTrainer class")
        try:
            self.gcloud.sync_folder_to_gcloud(self.model_pusher_config.bucket_name,
                                              self.model_pusher_config.trained_model_path,
                                              self.model_pusher_config.model_name)
            logging.info("Uploaded best model to gcloud storage")

             # Saving the model pusher artifacts
            model_pusher_artifact = ModelPusherArtifacts(
                bucket_name=self.model_pusher_config.bucket_name
            )
            logging.info("Exited the initiate_model_pusher method of ModelTrainer class")
            return model_pusher_artifact

        except Exception as e:
            raise InsuranceException(e, sys) from e
        
