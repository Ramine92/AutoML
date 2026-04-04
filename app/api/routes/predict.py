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

@router.post("/predict/json", summary="Predict using JSON", description="""
Send a JSON body with:
- **model_name**: name of the trained model (returned by /run)
- **data**: list of objects, each containing the feature columns from your CSV (without the target column)
""")
def predict_json(payload: dict = Body(...,example={"model_name":"LogisticRegression","data":[{"Age": 35, "Income": 65000, "Credit_Score": 710, "Loan_Amount": 30000, "Loan_Term": 24, "Employment_Status": "Employed"}]})):
    data = payload["data"]
    model_name = payload["model_name"]
    X_new = pd.DataFrame(data)
    prediction = predict(model_name,X_new)
    return PredictionResponse(prediction=prediction.tolist())
