import pandas as pd
from ml.data.data_manager import DataManager
from ml.preprocessing import Preprocessor 
from ml.models.registry import REGRESSION_MODELS,CLASSIFICATION_MODELS
class AutoMLPipeline:
    
    def __init__(self,target_column):
        self.target_column = target_column
        self.best_model_ = None # stores the best model 
        self.metrics_ = {} # stores metrics of each model
    
    def run(self,file_path):
        #loading and spliting data
        data_manager = DataManager(self.target_column)
        df,num_cols,cat_cols,isclassificaion = data_manager.load_and_profile(file_path=file_path)
        X_train,X_test,y_train,y_test = data_manager.get_split()
        #preprocessing data
        preprocessor = Preprocessor(cat_cols,num_cols)
        X_train_clean = preprocessor.fit_transform(X_train)
        X_test_clean = preprocessor.transform(X_test)
        #training model and evaluating best model
        if isclassificaion == False:
            score = -1
            for CLASSMODELS in REGRESSION_MODELS:
                model = CLASSMODELS()
                model.fit(X_train_clean,y_train)
                tmpscore = model.score(X_test_clean,y_test)
                if tmpscore["R2"] > score :
                    self.best_model_ = model
                    score = tmpscore["R2"]
                self.metrics_[CLASSMODELS.__name__] = tmpscore

        else:
            score = -1
            for CLASSMODELS in CLASSIFICATION_MODELS:
                model = CLASSMODELS()
                model.fit(X_train_clean,y_train)
                tmpscore = model.score(X_test_clean,y_test)
                if tmpscore["acc"] > score:
                    self.best_model_ = model
                    score = tmpscore["acc"]
                self.metrics_[CLASSMODELS.__name__] = tmpscore
        

    def predict(self,X_new):
        return self.best_model_.predict(X_new)




