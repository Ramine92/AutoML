from fastapi import APIRouter
from app.api.schemas import PredictionResponse
import pandas as pd
from app.services.predictor import predict
router = APIRouter()

@router.post("/predict")
async def make_prediction(model_name: str,data: list[dict]):
    X_new = pd.DataFrame(data)
    prediction = predict(model_name,X_new)
    return PredictionResponse(prediction=prediction.tolist())
