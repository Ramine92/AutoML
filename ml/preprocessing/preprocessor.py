from ml.preprocessing.imputer import Imputer
from ml.preprocessing.scaler import Scaler
from ml.preprocessing.encoder import Encoder
class Preprocessor:
    
    def __init__(self,cat_cols,num_cols):
        self.cat_cols = cat_cols
        self.num_cols = num_cols

        #instantiate transformers
        self.imputer = Imputer(self.cat_cols,self.num_cols)
        self.scaler = Scaler(self.num_cols)
        self.encoder = Encoder(self.cat_cols)


    def fit(self,X):
        imputed_X = self.imputer.fit_transform(X)

        scaled_X = self.scaler.fit_transform(imputed_X)

        self.encoder.fit_transform(scaled_X)

        return self

    def transform(self,X):
        X = self.imputer.transform(X)
        X = self.scaler.transform(X)
        X = self.encoder.transform(X)
        return X.astype(float)
    
    def fit_transform(self,X):
        return self.fit(X).transform(X)




