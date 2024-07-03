import os
import sys
import pandas as pd
import numpy as np
import dill
import yaml
from pandas import DataFrame

from insurance.exception import InsuranceException
from insurance.logger import logging
from insurance.constants import *

def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)

    except Exception as e:
        raise InsuranceException(e, sys) from e


def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise InsuranceException(e, sys) from e


def load_object(file_path: str) -> object:
    logging.info("Entered the load_object method of utils")

    try:

        with open(file_path, "rb") as file_obj:
            obj = dill.load(file_obj)

        logging.info("Exited the load_object method of utils")

        return obj

    except Exception as e:
        raise InsuranceException(e, sys) from e

def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise InsuranceException(e, sys) from e


def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise InsuranceException(e, sys) from e


def save_object(file_path: str, obj: object) -> None:
    logging.info("Entered the save_object method of utils")

    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

        logging.info("Exited the save_object method of utils")

    except Exception as e:
        raise InsuranceException(e, sys) from e


def drop_columns(df: DataFrame, cols: list)-> DataFrame:

    """
    drop the columns form a pandas DataFrame
    df: pandas DataFrame
    cols: list of columns to be dropped
    """
    logging.info("Entered drop_columns methon of utils")

    try:
        df = df.drop(columns=cols, axis=1)

        logging.info("Exited the drop_columns method of utils")
        
        return df
    except Exception as e:
        raise InsuranceException(e, sys) from e
    
def get_preprocessor_path(dir =ARTIFACT_DIR, t_dir = DATA_TRANSFORMATION_DIR_NAME, f_dir = DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR) -> str:
    try:
        timestamps= list(map(int, os.listdir(dir)))
        latest_timestamp= str(max(timestamps))
        latest_timestamp= '0'+latest_timestamp[0:1]+ '_'+ latest_timestamp[1:3] + '_' + latest_timestamp[3:7] + '_' + latest_timestamp[7:9] + '_' + latest_timestamp[9:11] + '_' + latest_timestamp[11:]
        latest_preprocessor_path= os.path.join(dir, f'{latest_timestamp}', t_dir, f_dir, PREPROCSSING_OBJECT_FILE_NAME)
        latest_timestamp= str(latest_timestamp).replace('\\','\\\\')
        return latest_preprocessor_path
    except Exception as e:
        raise InsuranceException(e, sys) from e
    
class CustomData:
    
    def __init__(self,
                 age: int,
                 sex: str,
                 bmi: float,
                 children: int,
                 smoker: str,
                 region: str):
        self.age = age
        self.sex= sex
        self.bmi= bmi
        self.children= children
        self.smoker= smoker
        self.region= region

    def get_data_as_data_frame(self):

            '''
            Return all input in the form of Dataframe. 
            
            '''
            try:
                custom_data_input_dict= {
                    "age": [self.age],
                    "sex": [self.sex],
                    "bmi": [self.bmi],
                    "children": [self.children],
                    "smoker": [self.smoker],
                    "region": [self.region],
                }

                return pd.DataFrame(custom_data_input_dict)

            except Exception as e:
                
                raise InsuranceException(e, sys) from e