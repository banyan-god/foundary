from fastapi import FastAPI
from api.health import router as health_router
from api.inference import router as inference_router
from api.training import router as training_router

from services.model_manager import load_all, save_all
app = FastAPI(
    title="Transaction Classifier API",
    description="Production-ready microservice for text classification",
    version="1.0.0"
)
app.include_router(health_router)
app.include_router(inference_router)
app.include_router(training_router)
@app.on_event("startup")
def on_startup():
    load_all()

@app.on_event("shutdown")
def on_shutdown():
    save_all()