import logging
from config import Config
from fastapi import FastAPI
# Configure root logger
level = getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO)
logging.basicConfig(level=level)
from api.health import router as health_router
from api.inference import router as inference_router
from api.training import router as training_router

from services.model_manager import load_all, save_all
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_all()
    yield
    save_all()

app = FastAPI(
    title="Transaction Classifier API",
    description="Production-ready microservice for text classification",
    version="1.0.0",
    lifespan=lifespan
)
app.include_router(health_router)
app.include_router(inference_router)
app.include_router(training_router)