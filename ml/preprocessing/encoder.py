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
        expected_dummy_cols = [dummy_col for cols in self.categories_.values() for dummy_col in cols]
        non_cat_cols = [c for c in X.columns if c not in expected_dummy_cols and c not in self.cat_cols]
        X = X.reindex(columns=non_cat_cols + expected_dummy_cols, fill_value=0)
        return X

    def fit_transform(self,X):
        return self.fit(X).transform(X)
    

