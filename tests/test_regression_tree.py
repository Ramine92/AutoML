from ml.data.data_manager import DataManager
from ml.models.Regression.RegressionTree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import numpy as np
from ml.preprocessing.preprocessor import Preprocessor
from app.core.config import BASE_DIR
target_column = "price"
data_path = BASE_DIR / "ml" / "data" / "tests" / "diamonds.csv"
def test_predict():
    #split data
    manager = DataManager(target_column)
    df,num_cols,cat_cols,isclassification = manager.load_and_profile(data_path)
    X,y = manager.get_split()
    print("Data Loaded")

    #preprocess data
    preprocessor = Preprocessor(cat_cols=cat_cols,num_cols=num_cols)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
    X_train_clean = preprocessor.fit_transform(X_train)
    X_test_clean = preprocessor.transform(X_test)
    print("Preprocessing Done")
    
    #train model
    X_train_small = X_train_clean[:200]
    y_train_small = y_train[:200]
    model = DecisionTreeRegressor(max_depth=5)
    model.fit(X_train_small, y_train_small, prune=False)
    score = model.score(X_test_clean,y_test)
    print("Model Trained")
    assert  (0 <= score["R2"] <= 1.0)

def test_best_split():
    X = np.array([[1],[2],[3],[7],[8],[9]])
    y = np.array([10,10,10,50,50,50])
    tree = DecisionTreeRegressor()
    best_feat,best_t = tree._best_split(X,y)
    assert best_feat == 0
    assert 3 <= best_t <= 7

def test_pruning():
    np.random.seed(42)
    X = np.linspace(0,10,20).reshape(-1,1)
    y = np.sin(X.flatten())+np.random.normal(0,0.1,20)

    tree = DecisionTreeRegressor(max_depth=10,min_samples_split=2)
    tree.fit(X,y,prune=False)

    leaves_before = tree._leaf_count(tree.root)

    tree._prune_with_alpha(tree.root,alpha=100.0)

    leaves_after = tree._leaf_count(tree.root)

    assert leaves_after < leaves_before
    assert leaves_after == 1
    







