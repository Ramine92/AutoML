from fastapi import FastAPI
from app.api.routes import health,run,predict

app = FastAPI(title="Fastapi autoML")


@app.get("/")
def root():
    return {"message":"Welcome to AutoML Project"}
app.include_router(router=health.router)
app.include_router(router=run.router)
app.include_router(router=predict.router)