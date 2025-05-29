from fastapi import APIRouter
import services.model_manager as mgr

router = APIRouter(tags=["Utility"])

@router.get("/health")
def health():
    # Return dynamic model version from model_manager
    return {"status": "ok", "model_version": mgr.model_version}