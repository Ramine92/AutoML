import pandas as pd
from ml.data.data_manager import DataManager
from ml.preprocessing.preprocessor import Preprocessor 
from sklearn.model_selection import KFold,cross_val_score
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
        df,num_cols,cat_cols,isclassification = data_manager.load_and_profile(file_path=file_path)
        X,y = data_manager.get_split()
        k_fold = KFold(n_splits=5,shuffle=True,random_state=42)
        
        best_overall_score = -float("inf")
        best_model_class = None
        
        if isclassification == False:
            for CLASSMODELS in REGRESSION_MODELS:
                with mlflow.start_run(run_name=CLASSMODELS.__name__):
                        fold_r2_score = 0
                        fold_mse_score = 0
                        fold_mae_score = 0
                        fold_rmse_score = 0
                        for fold,(train_idx,val_idx) in enumerate(k_fold.split(X)):
                            X_train = X.iloc[train_idx]
                            y_train = y.iloc[train_idx]
                            X_val = X.iloc[val_idx]
                            y_val = y.iloc[val_idx]
                            
                            #preprocessing data
                            preprocessor = Preprocessor(cat_cols,num_cols)
                            X_train_clean = preprocessor.fit_transform(X_train)
                            X_val_clean = preprocessor.transform(X_val)
                            
                            model = CLASSMODELS()
                            model.fit(X_train_clean,y_train)
                            tmpscore = model.score(X_val_clean,y_val)
                            fold_r2_score += tmpscore["R2"]
                            fold_mse_score += tmpscore["MSE"]
                            fold_mae_score += tmpscore["MAE"]
                            fold_rmse_score += tmpscore["RMSE"]
                            
                        fold_avg_r2_score = fold_r2_score/5
                        fold_avg_mse_score = fold_mse_score/5
                        fold_avg_mae_score = fold_mae_score/5
                        fold_avg_rmse_score = fold_rmse_score/5
                        
                        if fold_avg_r2_score > best_overall_score :
                                best_model_class = CLASSMODELS
                                best_overall_score = fold_avg_r2_score
                                
                        self.metrics_[CLASSMODELS.__name__] = {"R2":fold_avg_r2_score,"MSE":fold_avg_mse_score,"MAE":fold_avg_mae_score,"RMSE":fold_avg_rmse_score}
                        mlflow.log_metric("cv_r2_score",fold_avg_r2_score)
                        mlflow.log_metric("cv_mse",fold_avg_mse_score)
                        mlflow.log_metric("cv_mae",fold_avg_mae_score)
                        mlflow.log_metric("cv_rmse",fold_avg_rmse_score)
        else:
            for CLASSMODELS in CLASSIFICATION_MODELS:
                with mlflow.start_run(run_name=CLASSMODELS.__name__):
                        fold_acc_score = 0
                        fold_precision_score = 0
                        fold_f1score_score = 0
                        fold_recall_score = 0
                        for fold,(train_idx,val_idx) in enumerate(k_fold.split(X)):
                                X_train = X.iloc[train_idx]
                                y_train = y.iloc[train_idx]
                                X_val = X.iloc[val_idx]
                                y_val = y.iloc[val_idx]
                                
                                #preprocessing data
                                preprocessor = Preprocessor(cat_cols,num_cols)
                                X_train_clean = preprocessor.fit_transform(X_train)
                                X_val_clean = preprocessor.transform(X_val)
                                
                                model = CLASSMODELS()
                                model.fit(X_train_clean,y_train)
                                tmp_score = model.score(X_val_clean,y_val)
                                fold_acc_score += tmp_score["acc"]
                                fold_recall_score += tmp_score["recall"]
                                fold_f1score_score += tmp_score["f1_score"]
                                fold_precision_score += tmp_score["precision"]
                                
                        avg_fold_acc_score = fold_acc_score/5
                        avg_fold_precision_score = fold_precision_score/5
                        avg_fold_recall_score = fold_recall_score /5
                        avg_fold_f1_score = fold_f1score_score /5
                        if avg_fold_f1_score > best_overall_score:
                                best_model_class = CLASSMODELS
                                best_overall_score = avg_fold_f1_score
                                
                        self.metrics_[CLASSMODELS.__name__] = {"acc":avg_fold_acc_score,"precision":avg_fold_precision_score,"recall":avg_fold_recall_score,"f1_score":avg_fold_f1_score}
                        mlflow.log_metric("cv_accuracy",avg_fold_acc_score)
                        mlflow.log_metric("cv_precision",avg_fold_precision_score)
                        mlflow.log_metric("cv_recall",avg_fold_recall_score)
                        mlflow.log_metric("cv_f1score",avg_fold_f1_score)
                        
        # Retrain best model on ALL data
        self.best_model_name_ = best_model_class.__name__
        self.preprocessor_ = Preprocessor(cat_cols,num_cols)
        X_clean = self.preprocessor_.fit_transform(X)
        self.best_model_ = best_model_class()
        self.best_model_.fit(X_clean, y)

    def predict(self,X_new):
        return self.best_model_.predict(X_new)




