import sys
from pandas import DataFrame
from sklearn.pipeline import Pipeline

from insurance.exception import InsuranceException
from insurance.logger import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from insurance.entity.artifact_entity import RegressionMetricArtifact 

class InsuranceModel:
    def __init__(self, preprocessing_object: Pipeline, trained_model_object: object):
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object
    
    def predict(self, dataframe: DataFrame) -> DataFrame:

        logging.info("Entered predict method of UTruckModel class")
        try:
            logging.info("Using the trained model to get predictions")

            transformed_feature = self.preprocessing_object.transform(dataframe)

            logging.info("Used the trained model to get predictions")
            return self.trained_model_object.predict(transformed_feature)
        except Exception as e:
            raise InsuranceException(e, sys) from e
        
    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"   
        
def get_regression_metric(y_true, y_pred) -> RegressionMetricArtifact:
    try:
        mae= mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        metric_artifact = RegressionMetricArtifact(mae = mae,
                                                       mse= mse,
                                                       r2_score= r2)
        
        return metric_artifact

    except Exception as e:
        raise InsuranceException(e, sys) from e



# class InsuranceModel:

#     def __init__(self, preprocessor, model):

#         self.preprocessor= preprocessor
#         self.model= model

#     def predict(self, x):
#         try:
#             x_transform= self.preprocessor.transform(x)
#             y_hat= self.model.predict(x_transform)
#             return y_hat
#         except Exception as e:
#             raise InsuranceException(e, sys)
