import pandas as pd 
from pathlib import Path
from sklearn.model_selection import train_test_split
# convert data to a panda DataFame
# handle missing data and clean uncessary columns
# split data into test and train datasets
# send test and train data to ---> pipeline.features (feature engineering)

class DataManager:
    def __init__(self,target_column):
        self.target_column = target_column

    def load_and_profile(self,file_path):

        if not Path(file_path).exists():
            raise Exception("couldn't find the data passed.")
        self.df = pd.read_csv(file_path)

        self.numerical_cols = [c for c in self.df.select_dtypes(exclude="object").columns if c != self.target_column]
        self.categorical_cols = [c for c in self.df.select_dtypes(include="object").columns if c != self.target_column]

        #check if its a Classification / Regression problem
        if self.target_column not in self.df.columns:
            raise Exception("the target column does not exist.")
        if self.df[self.target_column].nunique() <= 20:
            self.isclassification = True
        else:
            self.isclassification = False
            
        return self.df,self.numerical_cols,self.categorical_cols,self.isclassification

    def get_split(self):
        #split the data into X and y
        X = self.df.drop(columns=self.target_column)
        y = self.df[self.target_column]
        return X,y



        



