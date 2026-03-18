import pandas as pd

class Imputer:

    def __init__(self,cat_cols,num_cols):
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.statistics_ = {}


    #learn from data
    def fit(self,X):
        for col in self.num_cols:
            self.statistics_[col] = X[col].median()
        for col in self.cat_cols:
            self.statistics_[col] = X[col].mode()[0] #most frequent value

        return self

    def transform(self,X):
        X = X.copy()
        for col,value in self.statistics_.items():
            X[col] = X[col].fillna(value)
        return X
    
    def fit_transform(self,X):
        return self.fit(X).transform(X)
    
    