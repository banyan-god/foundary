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
from models.tokenizer import SPTokenizer
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
logger.info(f"Using device: {device}")

def get_model_path(version):
    if not os.path.exists(Config.MODEL_DIR):
        os.makedirs(Config.MODEL_DIR)
    return os.path.join(Config.MODEL_DIR, f"model_v{version}.pt")

def save_all():
    global model, tokenizer, label2idx
    if model is None or tokenizer is None:
        return
    torch.save(model.state_dict(), get_model_path(model_version))
    with open(Config.LABELS_PATH, 'w') as f:
        json.dump(label2idx, f)
    logger.info("Model, tokenizer, and labels saved.")

def load_all():
    global model, tokenizer, label2idx, idx2label
    print("Loading model, tokenizer, and labels...")
    # Load labels
    if os.path.exists(Config.LABELS_PATH):
        with open(Config.LABELS_PATH) as f:
            label2idx = json.load(f)
        idx2label = {int(v): k for k, v in label2idx.items()}
    else:
        label2idx = {}
        idx2label = {}
    # Initialize SentencePiece tokenizer: train if model missing
    sp_model_path = f"{Config.SP_MODEL_PREFIX}.model"
    try:
        if not os.path.exists(sp_model_path):
            if not Config.SP_TRAIN_DATA or not os.path.exists(Config.SP_TRAIN_DATA):
                raise RuntimeError("SentencePiece model not found and SP_TRAIN_DATA is missing or invalid.")
            import csv
            with open(Config.SP_TRAIN_DATA, newline='') as f:
                reader = csv.reader(f)
                next(reader, None)
                num_sentences = sum(1 for _ in reader)
            vocab_size = min(Config.SP_VOCAB_SIZE, num_sentences) if num_sentences > 0 else Config.SP_VOCAB_SIZE
            SPTokenizer.train(
                input_file=Config.SP_TRAIN_DATA,
                model_prefix=Config.SP_MODEL_PREFIX,
                vocab_size=vocab_size
            )
        tokenizer = SPTokenizer(sp_model_path)
    except Exception as sp_err:
        logger.warning(f"SentencePiece train/load failed: {sp_err}. Using dummy tokenizer.")
        class DummyTokenizer:
            def encode(self, text: str): return [0]
            def save(self, path_prefix: str): pass
            @property
            def sp(self):
                class P:
                    def get_piece_size(self_inner): return 1
                return P()
        tokenizer = DummyTokenizer()
    # Load or initialize model
    # Initialize or load model with fixed SP vocab size
    vocab_size = tokenizer.sp.get_piece_size()
    model_file = get_model_path(model_version)
    if os.path.exists(model_file):
        try:
            state = torch.load(model_file, map_location=device)
            # infer number of classes from saved state or labels
            if 'fc.weight' in state:
                num_classes = state['fc.weight'].size(0)
            else:
                num_classes = len(label2idx) or 1
            model = VanillaTransformerClassifier(
                vocab_size=vocab_size,
                num_classes=num_classes
            ).to(device)
            model.load_state_dict(state)
            print(f"Loaded model version {model_version} with {num_classes} classes.")
        except Exception as exc:
            logger.warning(f"Could not load model from {model_file}: {exc}. Initializing new model.")
            model = VanillaTransformerClassifier(
                vocab_size=vocab_size,
                num_classes=max(len(label2idx), 3)
            ).to(device)
            print(f"Initialized new model with {len(label2idx)} classes.")
    else:
        model = VanillaTransformerClassifier(
            vocab_size=vocab_size,
            num_classes=max(len(label2idx), 3)
        ).to(device)
        print(f"Initialized new model with {len(label2idx)} classes.")
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
        # pad token sequences for batch
        seqs = [tokenizer.encode(t) for t in texts]
        max_len = max((len(s) for s in seqs), default=0)
        padded = [s + [0] * (max_len - len(s)) for s in seqs]
        input_ids = torch.tensor(padded, dtype=torch.long).to(device)
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
        # Fixed vocabulary from SentencePiece
        vocab_size = tokenizer.sp.get_piece_size()
        model = VanillaTransformerClassifier(
            vocab_size=vocab_size,
            num_classes=len(label2idx)
        ).to(device)
        # pad token sequences for training batch
        seqs = [tokenizer.encode(t) for t in texts]
        max_len = max((len(s) for s in seqs), default=0)
        padded = [s + [0] * (max_len - len(s)) for s in seqs]
        input_ids = torch.tensor(padded, dtype=torch.long).to(device)
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
        # No vocab growth; fixed SentencePiece vocabulary
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
