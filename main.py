import logging
from fastapi import FastAPI, HTTPException, Request, Header, status, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from typing import Optional, List
import torch
import torch.nn.functional as F
from model import (
    SimpleTokenizer, VanillaTransformerClassifier, 
    prepare_input_json, save_model, load_model,
    train_on_batch, online_learn_step, save_labels, load_labels
)
from schemas import (
    InferenceRequest, InferenceResponse, 
    TrainRequest, TrainResponse, OnlineLearnRequest, OnlineLearnResponse,
    BatchInferenceRequest, BatchInferenceResponse, DriftAlert
)
from config import Config
import os
import threading
import signal
import time
import json

# Thread safety
model_lock = threading.Lock()

# Logging setup
logger = logging.getLogger("transaction_classifier")
logger.setLevel(Config.LOG_LEVEL)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
console = logging.StreamHandler()
console.setFormatter(formatter)
logger.addHandler(console)
if not os.path.exists("logs"):
    os.makedirs("logs")
file_handler = logging.FileHandler("logs/app.log")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Model/Tokenizer/Labels
device = torch.device("cuda" if (Config.USE_CUDA and torch.cuda.is_available()) else "cpu")
tokenizer = None
model = None
label2idx = {}
idx2label = {}
model_version = Config.MODEL_VERSION

def get_model_path(version):
    if not os.path.exists(Config.MODEL_DIR):
        os.makedirs(Config.MODEL_DIR)
    return os.path.join(Config.MODEL_DIR, f"model_v{version}.pt")

def save_all():
    save_model(model, get_model_path(model_version))
    tokenizer.save(Config.TOKENIZER_PATH)
    save_labels(label2idx, Config.LABELS_PATH)
    logger.info("Model, tokenizer, and labels saved.")

def load_all():
    global tokenizer, model, label2idx, idx2label, model_version
    # Load labels
    if os.path.exists(Config.LABELS_PATH):
        label2idx, idx2label = load_labels(Config.LABELS_PATH)
        idx2label = {int(k): v for k, v in idx2label.items()}
    else:
        label2idx = {}
        idx2label = {}
    # Load tokenizer
    if os.path.exists(Config.TOKENIZER_PATH):
        tokenizer = SimpleTokenizer.load(Config.TOKENIZER_PATH)
    else:
        tokenizer = SimpleTokenizer()
    # Load model
    if os.path.exists(get_model_path(model_version)):
        model = VanillaTransformerClassifier(
            vocab_size=len(tokenizer.vocab),
            num_classes=len(label2idx)
        ).to(device)
        load_model(model, get_model_path(model_version), device)
    else:
        # Dummy model for cold start
        model = VanillaTransformerClassifier(
            vocab_size=max(len(tokenizer.vocab), 10),
            num_classes=max(len(label2idx), 3)
        ).to(device)
    logger.info("Model, tokenizer, and labels loaded.")

# Security: API Key
def api_key_auth(x_api_key: Optional[str] = Header(None)):
    if Config.API_KEY and x_api_key != Config.API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key")

# Data drift monitoring
def log_input_distribution(requests: List[InferenceRequest]):
    # Simple drift: log mean/variance of amount
    amounts = []
    for req in requests:
        try:
            amounts.append(float(req.current_transaction.amount))
        except Exception:
            continue
    if amounts:
        mean_amt = sum(amounts) / len(amounts)
        var_amt = sum((x - mean_amt) ** 2 for x in amounts) / len(amounts)
        logger.info(f"Input amount mean: {mean_amt:.2f}, variance: {var_amt:.2f}")
        # Simple drift alert
        if mean_amt > 10000 or var_amt > 1e7:
            logger.warning("Potential data drift detected in input amounts.")

# Graceful shutdown
def handle_shutdown(signum, frame):
    logger.info(f"Received signal {signum}, saving model/tokenizer/labels and shutting down.")
    save_all()
    exit(0)

signal.signal(signal.SIGTERM, handle_shutdown)
signal.signal(signal.SIGINT, handle_shutdown)

app = FastAPI(
    title="Transaction Classifier API",
    description="A robust, production-ready microservice for text classification with online/batch training, batch inference, and monitoring.",
    version="1.0.0"
)


@app.on_event("startup")
def startup_event():
    load_all()
    logger.info("Service started.")

@app.on_event("shutdown")
def shutdown_event():
    save_all()
    logger.info("Service stopped and model/tokenizer/labels saved.")

@app.get("/health", tags=["Utility"])
def health():
    return {"status": "ok", "model_version": model_version}

@app.post("/predict", response_model=InferenceResponse, tags=["Inference"])
def predict_endpoint(request: InferenceRequest, x_api_key: Optional[str] = Depends(api_key_auth)):
    try:
        with model_lock:
            model.eval()
            text = prepare_input_json(request.dict())
            input_ids = torch.tensor([tokenizer.encode(text)], dtype=torch.long).to(device)
            with torch.no_grad():
                logits = model(input_ids)
                probs = F.softmax(logits, dim=-1)
                pred_idx = torch.argmax(probs, dim=-1).item()
                pred_label = idx2label.get(pred_idx, "unknown")
                confidence = probs[0, pred_idx].item()
        return InferenceResponse(predicted_category=pred_label, confidence=confidence)
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/batch_predict", response_model=BatchInferenceResponse, tags=["Inference"])
def batch_predict_endpoint(request: BatchInferenceRequest, x_api_key: Optional[str] = Depends(api_key_auth)):
    if len(request.requests) > Config.MAX_BATCH_SIZE:
        raise HTTPException(status_code=400, detail=f"Batch size exceeds max of {Config.MAX_BATCH_SIZE}")
    try:
        with model_lock:
            model.eval()
            texts = [prepare_input_json(r.dict()) for r in request.requests]
            input_ids = torch.tensor([tokenizer.encode(t) for t in texts], dtype=torch.long).to(device)
            with torch.no_grad():
                logits = model(input_ids)
                probs = F.softmax(logits, dim=-1)
                pred_idxs = torch.argmax(probs, dim=-1).tolist()
                confidences = [probs[i, idx].item() for i, idx in enumerate(pred_idxs)]
                pred_labels = [idx2label.get(idx, "unknown") for idx in pred_idxs]
        results = [InferenceResponse(predicted_category=label, confidence=conf) for label, conf in zip(pred_labels, confidences)]
        log_input_distribution(request.requests)
        return BatchInferenceResponse(results=results)
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.post("/train", response_model=TrainResponse, tags=["Training"])
def train_endpoint(request: TrainRequest, x_api_key: Optional[str] = Depends(api_key_auth)):
    try:
        with model_lock:
            # Dynamic label set
            global tokenizer, model, label2idx, idx2label, model_version
            new_labels = set(request.labels)
            if not label2idx:
                label2idx = {label: i for i, label in enumerate(sorted(new_labels))}
            else:
                for label in new_labels:
                    if label not in label2idx:
                        label2idx[label] = len(label2idx)
            idx2label = {v: k for k, v in label2idx.items()}
            # Tokenizer growth
            texts = [prepare_input_json(x.dict()) for x in request.data]
            tokenizer.grow_vocab(texts)
            # Rebuild model if vocab or label set grew
            model = VanillaTransformerClassifier(
                vocab_size=len(tokenizer.vocab),
                num_classes=len(label2idx)
            ).to(device)
            # Train
            loss, acc = train_on_batch(
                model, tokenizer, [x.dict() for x in request.data], request.labels, label2idx,
                device
            )
            model_version += 1
            save_all()
            logger.info(f"Batch training complete. Loss: {loss:.4f}, Accuracy: {acc:.4f}")
        return TrainResponse(status="training_complete", loss=loss, accuracy=acc)
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")

@app.post("/online_learn", response_model=OnlineLearnResponse, tags=["Training"])
def online_learn_endpoint(request: OnlineLearnRequest, x_api_key: Optional[str] = Depends(api_key_auth)):
    try:
        with model_lock:
            global model, tokenizer, label2idx, idx2label, model_version
            # Dynamic label set
            if request.label not in label2idx:
                label2idx[request.label] = len(label2idx)
                idx2label = {v: k for k, v in label2idx.items()}
                # Rebuild model if label set grew
                model = VanillaTransformerClassifier(
                    vocab_size=len(tokenizer.vocab),
                    num_classes=len(label2idx)
                ).to(device)
            # Tokenizer growth
            text = prepare_input_json(request.input.dict())
            tokenizer.grow_vocab([text])
            # Rebuild model if vocab grew
            model = VanillaTransformerClassifier(
                vocab_size=len(tokenizer.vocab),
                num_classes=len(label2idx)
            ).to(device)
            # Online learn
            loss = online_learn_step(
                model, tokenizer, request.input.dict(), request.label, label2idx, device
            )
            model_version += 1
            save_all()
            logger.info(f"Online learning step complete. Loss: {loss:.4f}")
        return OnlineLearnResponse(status="online_learning_complete", loss=loss)
    except Exception as e:
        logger.error(f"Online learning error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Online learning error: {str(e)}")

@app.get("/docs", include_in_schema=False)
def custom_docs():
    return get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

@app.get("/", include_in_schema=False)
def root():
    return {"message": "Welcome to the Transaction Classifier API. See /docs for usage."}
