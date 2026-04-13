from fastapi import APIRouter,UploadFile,Form,BackgroundTasks
from app.api.schemas import ModelResult,RunResponse
from app.services import predictor
import asyncio

router = APIRouter()
jobs = {}

@router.post("/run", summary="Train models and return the best one", description="""
Upload a CSV file and specify the target column name.
The pipeline will:
1. **Detect** if it's classification or regression
2. **Preprocess** the data automatically
3. **Train** all candidate models
4. **Return** metrics for each model and the best one
""")
async def run(file: UploadFile,target_column: str= Form(...),background_tasks: BackgroundTasks = None):
   import uuid
   job_id = str(uuid.uuid4())
   path = await predictor.save_path(file)
   jobs[job_id] = {"status":"running","result":None}
   background_tasks.add_task(run_training,job_id,path,target_column)
   return {"job_id":job_id}

def run_training(job_id,path,target_column):
   try:
      best_model_name,run_metrics =predictor.start_run(path=path,target_column=target_column)
      # Format results for Streamlit compatibility
      results = [{"model": name, "metrics": scores} for name, scores in run_metrics.items()]
      jobs[job_id] = {"status":"done","best_model":best_model_name,"results":results}
   except Exception as e:
      jobs[job_id] = {"status":"failed","error":str(e)}

@router.get("/status/{job_id}")
def get_status(job_id: str):
   return jobs.get(job_id,{"status": "not_found"})

