import pandas as pd
from ml.data.data_manager import DataManager
from ml.preprocessing import Preprocessor 
from ml.models.registry import REGRESSION_MODELS,CLASSIFICATION_MODELS
import mlflow
class AutoMLPipeline:
    def __init__(self,target_column):
        self.target_column = target_column
        self.best_model_ = None # stores the best model
        self.preprocessor_ = None
        self.best_model_name_ = None # stores the best model name
        self.metrics_ = {} # stores metrics of each model
    
    def run(self,file_path):
        mlflow.set_experiment(f"Predict {self.target_column}")
        #loading and spliting data
        data_manager = DataManager(self.target_column)
        df,num_cols,cat_cols,isclassificaion = data_manager.load_and_profile(file_path=file_path)
        X_train,X_test,y_train,y_test = data_manager.get_split()
        #preprocessing data
        preprocessor = Preprocessor(cat_cols,num_cols)
        self.preprocessor_ = preprocessor
        X_train_clean = preprocessor.fit_transform(X_train)
        X_test_clean = preprocessor.transform(X_test)
        #training model and evaluating best model
        if isclassificaion == False:
            score = -1
            for CLASSMODELS in REGRESSION_MODELS:
                with mlflow.start_run(run_name=CLASSMODELS.__name__):
                    model = CLASSMODELS()
                    model.fit(X_train_clean,y_train)
                    tmpscore = model.score(X_test_clean,y_test)
                    if tmpscore["R2"] > score :
                        self.best_model_ = model
                        self.best_model_name_ = CLASSMODELS.__name__
                        score = tmpscore["R2"]
                    self.metrics_[CLASSMODELS.__name__] = tmpscore
                    mlflow.log_metric("r2_score",tmpscore["R2"])
                    mlflow.log_metric("mse",tmpscore["MSE"])
                    mlflow.log_metric("mae",tmpscore["MAE"])
                    mlflow.log_metric("rmse",tmpscore["RMSE"])
        else:
            score = -1
            for CLASSMODELS in CLASSIFICATION_MODELS:
                with mlflow.start_run(run_name=CLASSMODELS.__name__):
                    model = CLASSMODELS()
                    model.fit(X_train_clean,y_train)
                    tmpscore = model.score(X_test_clean,y_test)
                    if tmpscore["acc"] > score:
                        self.best_model_ = model
                        self.best_model_name_ = CLASSMODELS.__name__
                        score = tmpscore["acc"]
                    self.metrics_[CLASSMODELS.__name__] = tmpscore
                    mlflow.log_metric("accuracy",tmpscore["acc"])

    def predict(self,X_new):
        return self.best_model_.predict(X_new)




