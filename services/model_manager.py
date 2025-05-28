import os
import json
import logging
import threading
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from config import Config
from models.transformer import VanillaTransformerClassifier
from models.tokenizer import SimpleTokenizer
from schemas.transaction import (
    InferenceRequest, InferenceResponse,
    BatchInferenceRequest, BatchInferenceResponse,
    TrainRequest, TrainResponse,
    OnlineLearnRequest, OnlineLearnResponse
)

# Thread-safe model management
model_lock = threading.Lock()
# Select best available device: CUDA > MPS (macOS) > CPU
_use_cuda = Config.USE_CUDA and torch.cuda.is_available()
_use_mps = False
try:
    _use_mps = torch.backends.mps.is_available() and torch.backends.mps.is_built()
except Exception:
    _use_mps = False
device = torch.device("cuda" if _use_cuda else "mps" if _use_mps else "cpu")
model = None
tokenizer = None
label2idx = {}
idx2label = {}
model_version = Config.MODEL_VERSION
logger = logging.getLogger("transaction_classifier")

def get_model_path(version):
    if not os.path.exists(Config.MODEL_DIR):
        os.makedirs(Config.MODEL_DIR)
    return os.path.join(Config.MODEL_DIR, f"model_v{version}.pt")

def save_all():
    global model, tokenizer, label2idx
    if model is None or tokenizer is None:
        return
    torch.save(model.state_dict(), get_model_path(model_version))
    tokenizer.save(Config.TOKENIZER_PATH)
    with open(Config.LABELS_PATH, 'w') as f:
        json.dump(label2idx, f)
    logger.info("Model, tokenizer, and labels saved.")

def load_all():
    global model, tokenizer, label2idx, idx2label
    # Load labels
    if os.path.exists(Config.LABELS_PATH):
        with open(Config.LABELS_PATH) as f:
            label2idx = json.load(f)
        idx2label = {int(v): k for k, v in label2idx.items()}
    else:
        label2idx = {}
        idx2label = {}
    # Load tokenizer
    if os.path.exists(Config.TOKENIZER_PATH):
        tokenizer = SimpleTokenizer.load(Config.TOKENIZER_PATH)
    else:
        tokenizer = SimpleTokenizer()
    # Load or initialize model
    if os.path.exists(get_model_path(model_version)):
        model = VanillaTransformerClassifier(
            vocab_size=len(tokenizer.vocab),
            num_classes=len(label2idx)
        ).to(device)
        state = torch.load(get_model_path(model_version), map_location=device)
        model.load_state_dict(state)
    else:
        model = VanillaTransformerClassifier(
            vocab_size=max(len(tokenizer.vocab), 10),
            num_classes=max(len(label2idx), 3)
        ).to(device)
    logger.info("Model, tokenizer, and labels loaded.")
#    return
#    return

def prepare_input_json(req_dict: dict) -> str:
    parts = []
    for hist in req_dict.get('user_history', []):
        parts.append(f"{hist['description']} {hist['name']} {hist['merchant']} {hist['amount']} {hist.get('category', '')}")
    cur = req_dict['current_transaction']
    parts.append(f"{cur['description']} {cur['name']} {cur['merchant']} {cur['amount']}")
    return ' '.join(parts)

def log_input_distribution(requests):
    amounts = []
    for req in requests:
        try:
            amounts.append(float(req.current_transaction.amount))
        except Exception:
            continue
    if amounts:
        mean_amt = sum(amounts) / len(amounts)
        var_amt = sum((x - mean_amt)**2 for x in amounts) / len(amounts)
        logger.info(f"Input amount mean: {mean_amt:.2f}, variance: {var_amt:.2f}")
        if mean_amt > 10000 or var_amt > 1e7:
            logger.warning("Potential data drift detected in input amounts.")

def predict(request: InferenceRequest) -> InferenceResponse:
    with model_lock:
        if model is None or tokenizer is None:
            load_all()
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

def batch_predict(request: BatchInferenceRequest) -> BatchInferenceResponse:
    with model_lock:
        if model is None or tokenizer is None:
            load_all()
        model.eval()
        texts = [prepare_input_json(r.dict()) for r in request.requests]
        input_ids = torch.tensor([tokenizer.encode(t) for t in texts], dtype=torch.long).to(device)
        with torch.no_grad():
            logits = model(input_ids)
            probs = F.softmax(logits, dim=-1)
            pred_idxs = torch.argmax(probs, dim=-1).tolist()
            confidences = [probs[i, idx].item() for i, idx in enumerate(pred_idxs)]
            pred_labels = [idx2label.get(idx, "unknown") for idx in pred_idxs]
        results = [InferenceResponse(predicted_category=lbl, confidence=conf)
                   for lbl, conf in zip(pred_labels, confidences)]
        log_input_distribution(request.requests)
        return BatchInferenceResponse(results=results)

def train(request: TrainRequest) -> TrainResponse:
    global model, tokenizer, label2idx, idx2label, model_version
    # Always (re)load model and tokenizer to ensure fresh state
    load_all()
    with model_lock:
        new_labels = set(request.labels)
        if not label2idx:
            label2idx = {label: i for i, label in enumerate(sorted(new_labels))}
        else:
            for lbl in new_labels:
                if lbl not in label2idx:
                    label2idx[lbl] = len(label2idx)
        idx2label = {v: k for k, v in label2idx.items()}
        texts = [prepare_input_json(x.dict()) for x in request.data]
        tokenizer.grow_vocab(texts)
        model = VanillaTransformerClassifier(
            vocab_size=len(tokenizer.vocab),
            num_classes=len(label2idx)
        ).to(device)
        input_ids = torch.tensor([tokenizer.encode(t) for t in texts], dtype=torch.long).to(device)
        labels_idx = torch.tensor([label2idx[lbl] for lbl in request.labels], dtype=torch.long).to(device)
        model.train()
        optimizer = optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()
        # Train for multiple epochs to fit small dataset
        epochs = 5
        for _ in range(epochs):
            logits = model(input_ids)
            loss = criterion(logits, labels_idx)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Evaluate
        with torch.no_grad():
            logits = model(input_ids)
            preds = torch.argmax(logits, dim=-1)
        accuracy = (preds == labels_idx).float().mean().item()
        model_version += 1
        save_all()
        return TrainResponse(status="training_complete", loss=loss.item(), accuracy=accuracy)

def online_learn(request: OnlineLearnRequest) -> OnlineLearnResponse:
    # Always (re)load model and tokenizer before online learning
    load_all()
    with model_lock:
        text = prepare_input_json(request.input.dict())
        tokenizer.grow_vocab([text])
        input_ids = torch.tensor([tokenizer.encode(text)], dtype=torch.long).to(device)
        label_idx = torch.tensor([label2idx.get(request.label, 0)], dtype=torch.long).to(device)
        model.train()
        optimizer = optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()
        logits = model(input_ids)
        loss = criterion(logits, label_idx)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        save_all()
        return OnlineLearnResponse(status="online_learning_complete", loss=loss.item())
