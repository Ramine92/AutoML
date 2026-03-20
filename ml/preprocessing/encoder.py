import pandas as pd

class Encoder:
    def __init__(self,cat_cols):
        self.cat_cols = cat_cols
        self.categories_ = {}

    def fit(self,X):
        for col in self.cat_cols:
            col_tmp = pd.get_dummies(data=X[col],prefix=col)
            self.categories_[col] = col_tmp.columns.tolist()
        
        return self
    
    def transform(self,X):
        X = X.copy()
        X = pd.get_dummies(data=X,columns=self.cat_cols)
        expected_cols = [col for cols in self.categories_[col].values() for col in self.cat_cols]
        X = X.reindex(columns=expected_cols,fill_value=0)
        return X

    def fit_transform(self,X):
        return self.fit(X).transform(X)
    

