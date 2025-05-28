from fastapi import APIRouter
from services.model_manager import model_version

router = APIRouter(tags=["Utility"])

@router.get("/health")
def health():
    return {"status": "ok", "model_version": model_version}