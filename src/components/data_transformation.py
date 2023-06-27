from src.exception import SensorException
from src.logger import logging
import os,sys 
from sklearn.pipeline import Pipeline
import pandas as pd
from src.utils import save_object
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.combine import SMOTETomek
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self)->Pipeline:
        try:
            simple_imputer = SimpleImputer(strategy='constant', fill_value=0)
            robust_scaler =  RobustScaler()
            pipeline = Pipeline(steps=[
                    ('Imputer',simple_imputer),
                    ('RobustScaler',robust_scaler)
                ])
            return pipeline
        except Exception as e:
            raise SensorException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            #reading training and testing file
            logging.info("Reading training and testing File")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            TARGET_COLUMN = "class"
            
            logging.info("Selecting input feature for train and test dataframe")
            input_feature_train_df=train_df.drop(TARGET_COLUMN,axis=1)
            input_feature_test_df=test_df.drop(TARGET_COLUMN,axis=1)

            logging.info("Selecting target feature for train and test dataframe")
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_test_df = test_df[TARGET_COLUMN]

            label_encoder = LabelEncoder()
            label_encoder.fit(target_feature_train_df)

            logging.info("Transformation on target columns")
            target_feature_train_arr = label_encoder.transform(target_feature_train_df)
            target_feature_test_arr = label_encoder.transform(target_feature_test_df)
            

            transformation_pipleine = self.get_data_transformer_object()
            transformation_pipleine.fit(input_feature_train_df)

            logging.info("Transforming input features")
            input_feature_train_arr = transformation_pipleine.transform(input_feature_train_df)
            input_feature_test_arr = transformation_pipleine.transform(input_feature_test_df)


            smt = SMOTETomek(random_state = 42)
            logging.info(f"Before resampling in training set Input: {input_feature_train_arr.shape} Target:{target_feature_train_arr.shape}")
            input_feature_train_arr, target_feature_train_arr = smt.fit_resample(input_feature_train_arr, target_feature_train_arr)
            logging.info(f"After resampling in training set Input: {input_feature_train_arr.shape} Target:{target_feature_train_arr.shape}")

            logging.info(f"Before resampling in testing set Input: {input_feature_test_arr.shape} Target:{target_feature_test_arr.shape}")
            input_feature_test_arr, target_feature_test_arr = smt.fit_resample(input_feature_test_arr, target_feature_test_arr)
            logging.info(f"After resampling in testing set Input: {input_feature_test_arr.shape} Target:{target_feature_test_arr.shape}")

            #target encoder
            train_arr = np.c_[input_feature_train_arr, target_feature_train_arr ]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_arr]

            logging.info("Data transformation completed")
            
            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path,
             obj=transformation_pipleine)

            return (train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path)
        
        except Exception as e:
            raise SensorException(e, sys)
