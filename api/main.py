from fastapi import FastAPI
from api.health import router as health_router
from api.inference import router as inference_router
from api.training import router as training_router

app = FastAPI(
    title="Transaction Classifier API",
    description="Production-ready microservice for text classification",
    version="1.0.0"
)
app.include_router(health_router)
app.include_router(inference_router)
app.include_router(training_router)