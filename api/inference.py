from fastapi import APIRouter, HTTPException, status
from config import Config
from services.model_manager import predict, batch_predict
from schemas.transaction import InferenceRequest, BatchInferenceRequest, InferenceResponse, BatchInferenceResponse

router = APIRouter(prefix="", tags=["Inference"])

@router.post("/predict", response_model=InferenceResponse)
def predict_endpoint(request: InferenceRequest):
    try:
        return predict(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch_predict", response_model=BatchInferenceResponse)
def batch_predict_endpoint(request: BatchInferenceRequest):
    if len(request.requests) > Config.MAX_BATCH_SIZE:
        raise HTTPException(status_code=400,
                            detail=f"Batch size exceeds max of {Config.MAX_BATCH_SIZE}")
    try:
        return batch_predict(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))