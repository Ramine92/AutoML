from fastapi import APIRouter,UploadFile
from app.api.schemas import ModelResult,RunResponse
from app.services import predictor

router = APIRouter()

@router.post("/run")
async def run(file: UploadFile,target_column: str):
   path = await predictor.save_path(file)
   best_model_name,run_metrics = predictor.start_run(path=path,target_column=target_column)
   results = [ModelResult(model=name,metrics=scores) for name,scores in run_metrics.items()]
   return RunResponse(best_model=best_model_name,results=results)

