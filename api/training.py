from fastapi import APIRouter, HTTPException
from config import Config
from services.model_manager import train, online_learn
from schemas.transaction import TrainRequest, TrainResponse, OnlineLearnRequest, OnlineLearnResponse

router = APIRouter(prefix="", tags=["Training"])

@router.post("/train", response_model=TrainResponse)
def train_endpoint(request: TrainRequest):
    if not request.data or not request.labels:
        raise HTTPException(status_code=400, detail="data and labels must be provided")
    try:
        return train(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/online_learn", response_model=OnlineLearnResponse)
def online_learn_endpoint(request: OnlineLearnRequest):
    try:
        return online_learn(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))