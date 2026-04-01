from fastapi import APIRouter,Form,UploadFile,Body
from app.api.schemas import PredictionResponse
import pandas as pd
from app.services.predictor import predict
router = APIRouter()

@router.post("/predict/file")
def predict_file(data: UploadFile,model_name: str = Form(...)):
    data = pd.read_csv(data.file)
    X_new = pd.DataFrame(data)
    prediction = predict(model_name,X_new)
    return PredictionResponse(prediction=prediction.tolist())

@router.post("/predict/json")
def predict_json(payload: dict = Body(...)):
    data = payload["data"]
    model_name = payload["model_name"]
    X_new = pd.DataFrame(data)
    prediction = predict(model_name,X_new)
    return PredictionResponse(prediction=prediction.tolist())
