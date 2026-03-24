# pipeline_test.py — sequential smoke test
from ml.data.data_manager import DataManager
from ml.preprocessing.preprocessor import Preprocessor
from ml.models.classification.LogisticRegression import LogisticRegression
from ml.models.Regression.LinearRegression import LinearRegression
from app.core.config import BASE_DIR

target_column = "Loan_Approved"
data_path = BASE_DIR / "ml" / "data" / "tests" / "loan_prediction_dataset.csv"
def test_pipeline():

    # Step 1: Load data
    manager = DataManager(target_column=target_column)
    df, num_cols, cat_cols, is_classification = manager.load_and_profile(data_path)
    X_train, X_test, y_train, y_test = manager.get_split()
    print("Data loaded:", X_train.shape, X_test.shape)

    assert (len(X_train) > 80) and (len(X_test) > 20)

    # Step 2: Preprocess
    preprocessor = Preprocessor(cat_cols=cat_cols, num_cols=num_cols)
    X_train_clean = preprocessor.fit_transform(X_train)
    X_test_clean  = preprocessor.transform(X_test)
    print("Preprocessing done:", X_train_clean.shape)

    # Step 3: Train model
    model = LogisticRegression() if is_classification else LinearRegression()
    model.fit(X_train_clean, y_train)
    print("Model trained")

    # Step 4: Score 

    score = model.score(X_test_clean,y_test)
    print(f"Model score is : {score}")
    assert  0.0 <= score <= 1.0
