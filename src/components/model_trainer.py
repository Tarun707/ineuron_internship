from src.exception import SensorException
from src.logger import logging
import os,sys 
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from src.utils import save_object, evaluate_models
from sklearn.metrics import f1_score
from dataclasses import dataclass
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    # def fine_tune(self):
    #     try:
    #         #Wite code for Grid Search CV
    #         pass


    #     except Exception as e:
    #         raise SensorException(e, sys)

    # def train_model(self,x,y):
    #     try:
    #         xgb_clf =  XGBClassifier()
    #         xgb_clf.fit(x,y)
    #         return xgb_clf
    #     except Exception as e:
    #         raise SensorException(e, sys)


    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info(f"Splitting input and target feature from both train and test arr.")
            X_train,y_train = train_arr[:,:-1],train_arr[:,-1]
            X_test,y_test = test_arr[:,:-1],test_arr[:,-1]

            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "Logitic Regression": LogisticRegression(max_iter = 200),
                "K Neighbors Classifier": KNeighborsClassifier(),
                "XGBClassifier": XGBClassifier(),
                "AdaBoost Regressor": AdaBoostClassifier(),
            }

            logging.info("Evaluation of Models started")
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test, models=models)

            logging.info("Getting best model score")
            best_model_score = max(list(model_report.values()))

            logging.info("Getting best model name")
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise SensorException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            logging.info("Model Training completed")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            f1 = f1_score(y_test, predicted)
            return (f1, best_model_name)

            # logging.info(f"Checking if our model is underfitting or not")
            # if f1_test_score<self.model_trainer_config.expected_score:
            #     raise Exception(f"Model is not good as it is not able to give \
            #     expected accuracy: {self.model_trainer_config.expected_score}: model actual score: {f1_test_score}")

            # logging.info(f"Checking if our model is overfiiting or not")
            # diff = abs(f1_train_score-f1_test_score)

            # if diff>self.model_trainer_config.overfitting_threshold:
            #     raise Exception(f"Train and test score diff: {diff} is more than overfitting threshold {self.model_trainer_config.overfitting_threshold}")

        except Exception as e:
            raise SensorException(e, sys)
