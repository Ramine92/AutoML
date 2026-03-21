import pandas as pd
import numpy as np

class Scaler:
    def __init__(self,num_cols,method="standard"):
        self.num_cols = num_cols
        #used later in MinMaxScaler
        self.min_ = {}
        self.max_ = {} 
        #used later in StandardScaler
        self.mean_ = {}
        self.std_ = {}
        #strategy to use 
        self.method = method
    
    def fit(self,X):
        X = X.copy()
        self.m = len(X)
        for col in self.num_cols:
            if self.method == "minmax":
                self.min_[col] = X[col].min()
                self.max_[col] = X[col].max()
            else:
                self.mean_[col] = X[col].mean()
                self.std_[col] = np.sqrt((1/self.m)*np.sum((X[col]-self.mean_[col])**2))
        return self
    
    def transform(self,X):
        X = X.copy()
        if self.method == "minmax":
            for col in self.num_cols:
                X[col] = (X[col]-self.min_[col])/(self.max_[col]-self.min_[col])
        else:
            for col in self.num_cols:
                X[col] = (X[col]-self.mean_[col])/self.std_[col]
        return X
    
    def fit_transform(self,X):
        return self.fit(X).transform(X)
