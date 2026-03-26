from pydantic import BaseModel


class ModelResult(BaseModel):
    model:str
    metrics:dict

class RunResponse(BaseModel):
    best_model: str
    results: list[ModelResult]

class PredictionResponse(BaseModel):
    prediction: list
