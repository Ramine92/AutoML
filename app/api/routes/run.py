from fastapi import APIRouter,UploadFile,Form
from app.api.schemas import ModelResult,RunResponse
from app.services import predictor

router = APIRouter()

@router.post("/run", summary="Train models and return the best one", description="""
Upload a CSV file and specify the target column name.
The pipeline will:
1. **Detect** if it's classification or regression
2. **Preprocess** the data automatically
3. **Train** all candidate models
4. **Return** metrics for each model and the best one
""")
async def run(file: UploadFile,target_column: str= Form(...)):
   path = await predictor.save_path(file)
   best_model_name,run_metrics = predictor.start_run(path=path,target_column=target_column)
   results = [ModelResult(model=name,metrics=scores) for name,scores in run_metrics.items()]
   return RunResponse(best_model=best_model_name,results=results)

