from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict

class Transaction(BaseModel):
    description: str
    name: str
    merchant: str
    amount: str
    category: Optional[str] = None

    @field_validator("amount")
    def amount_must_be_float(cls, v):
        try:
            float(v)
        except Exception:
            raise ValueError("amount must be a float or string representing a float")
        return v

class UserHistoryItem(Transaction):
    category: str

class InferenceRequest(BaseModel):
    current_transaction: Transaction
    user_history: List[UserHistoryItem] = []

class InferenceResponse(BaseModel):
    predicted_category: str
    confidence: float

class BatchInferenceRequest(BaseModel):
    requests: List[InferenceRequest]

class BatchInferenceResponse(BaseModel):
    results: List[InferenceResponse]

class TrainRequest(BaseModel):
    data: List[InferenceRequest]
    labels: List[str]

class TrainResponse(BaseModel):
    status: str
    loss: float
    accuracy: float

class OnlineLearnRequest(BaseModel):
    input: InferenceRequest
    label: str

class OnlineLearnResponse(BaseModel):
    status: str
    loss: float

class DriftAlert(BaseModel):
    alert: str
    details: Dict