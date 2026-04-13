from ml.pipelines.auto_ml_pipeline import AutoMLPipeline
from fastapi import UploadFile
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
MODELS_DIR = BASE_DIR / "ml" / "data" / "artifacts"
UPLOAD_DIR = BASE_DIR / "ml" / "data" / "uploaded"
async def save_path(file: UploadFile):
    path = f"{UPLOAD_DIR}/{file.filename}"
    UPLOAD_DIR.mkdir(parents=True,exist_ok=True)
    with open(path,"wb") as f:
        content = await file.read()
        f.write(content)
    return path

def start_run(path: str,target_column: str):
    auto_ml = AutoMLPipeline(target_column=target_column)
    auto_ml.run(file_path=path)
    MODELS_DIR.mkdir(parents=True,exist_ok=True)
    model_path = MODELS_DIR / f"{auto_ml.best_model_name_}.pkl"
    preprocessor_path = MODELS_DIR / "preprocessor.pkl"
    joblib.dump(auto_ml.best_model_,model_path)
    joblib.dump(auto_ml.preprocessor_,preprocessor_path)
    return auto_ml.best_model_name_,auto_ml.metrics_

def predict(model_name,X_new):
    model_path = MODELS_DIR / f"{model_name}.pkl"
    preprocessor_path = MODELS_DIR / "preprocessor.pkl"
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    X_new_cleaned = preprocessor.transform(X_new)
    return model.predict(X_new_cleaned)





