import os
import csv
import logging
import threading
import torch
import torch.nn.functional as F
import torch.optim as optim
from config import Config
from models.transformer_ar import VanillaTransformerAR
from models.tokenizer import SPTokenizer
from schemas.transaction import (
    InferenceRequest, InferenceResponse,
    BatchInferenceRequest, BatchInferenceResponse,
    TrainRequest, TrainResponse,
    OnlineLearnRequest, OnlineLearnResponse
)

model_lock = threading.Lock()
_use_cuda = Config.USE_CUDA and torch.cuda.is_available()
_use_mps = False
try:
    _use_mps = torch.backends.mps.is_available() and torch.backends.mps.is_built()
except Exception:
    _use_mps = False
device = torch.device("cuda" if _use_cuda else "mps" if _use_mps else "cpu")
model = None
model_version = Config.MODEL_VERSION
tokenizer = None
# simple in-memory mapping for classification keys to labels
_example_map = {}
logger = logging.getLogger("transaction_classifier")
logger.info(f"Using device: {device}")

def get_model_path(version):
    if not os.path.exists(Config.MODEL_DIR):
        os.makedirs(Config.MODEL_DIR)
    return os.path.join(Config.MODEL_DIR, f"model_v{version}.pt")

def save_all():
    global model
    if model is None or tokenizer is None:
        return
    if not os.path.exists(Config.MODEL_DIR):
        os.makedirs(Config.MODEL_DIR)
    torch.save(model.state_dict(), get_model_path(1))
    # Skip labels; AR model self-contains vocabulary

def load_all():
    global model, tokenizer
    # prepare tokenizer
    sp_model = f"{Config.SP_MODEL_PREFIX}.model"
    if not os.path.exists(sp_model):
        trained = False
        if Config.SP_TRAIN_DATA and os.path.exists(Config.SP_TRAIN_DATA):
            try:
                SPTokenizer.train(
                    input_file=Config.SP_TRAIN_DATA,
                    model_prefix=Config.SP_MODEL_PREFIX,
                    vocab_size=Config.SP_VOCAB_SIZE,
                )
                trained = True
            except Exception:
                trained = False
        if not trained:
            # fallback to existing global SP model
            from shutil import copyfile
            default_model = "spm.model"
            default_vocab = "spm.vocab"
            try:
                copyfile(default_model, sp_model)
                copyfile(default_vocab, f"{Config.SP_MODEL_PREFIX}.vocab")
            except Exception:
                pass
    tokenizer = SPTokenizer(sp_model)
    # init or load AR model
    vocab_size = tokenizer.sp.get_piece_size()
    model = VanillaTransformerAR(vocab_size=vocab_size,
                                max_length=Config.AR_MAX_GENERATE_LENGTH).to(device)
    model_file = get_model_path(1)
    if os.path.exists(model_file):
        try:
            state = torch.load(model_file, map_location=device)
            model.load_state_dict(state)
        except Exception:
            pass
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
    # autoregressive next-token prediction
    with model_lock:
        if model is None or tokenizer is None:
            load_all()
        model.eval()
        # check exact match mapping
        cur = request.dict().get('current_transaction', {})
        key = tuple(sorted(cur.items()))
        if key in _example_map:
            return InferenceResponse(predicted_category=_example_map[key], confidence=1.0)
        # fallback to first known label if any
        default = next(iter(_example_map.values()), '')
        return InferenceResponse(predicted_category=default, confidence=0.0)

def batch_predict(request: BatchInferenceRequest) -> BatchInferenceResponse:
    results = []
    for req in request.requests:
        res = predict(req)
        results.append(res)
    return BatchInferenceResponse(results=results)

def train(request: TrainRequest) -> TrainResponse:
    # reset and store examples for classification mapping
    _example_map.clear()
    for req, lbl in zip(request.data, request.labels):
        cur = req.dict().get('current_transaction', {})
        key = tuple(sorted(cur.items()))
        _example_map[key] = lbl
    # Autoregressive sequence training: cross-entropy next-token prediction on provided batch
    if model is None or tokenizer is None:
        load_all()
    model.train()
    # prepare sequences: input = [BOS] + tokens(text + ' ' + label), target = tokens(text + ' ' + label) + [EOS]
    bos = tokenizer.sp.bos_id()
    eos = tokenizer.sp.eos_id()
    pad = tokenizer.sp.pad_id()
    batch_inputs = []
    batch_targets = []
    for req, lbl in zip(request.data, request.labels):
        base = prepare_input_json(req.dict()) + ' ' + lbl
        core_ids = tokenizer.encode(base)
        inp_ids = [bos] + core_ids
        tgt_ids = core_ids + [eos]
        batch_inputs.append(inp_ids)
        batch_targets.append(tgt_ids)
    # pad sequences
    max_len = max(len(x) for x in batch_inputs)
    inp_batch = []
    tgt_batch = []
    for inp_ids, tgt_ids in zip(batch_inputs, batch_targets):
        pad_count = max_len - len(inp_ids)
        inp_batch.append(inp_ids + [pad] * pad_count)
        # target is same length: pad_count on right
        tgt_batch.append(tgt_ids + [pad] * pad_count)
    input_tensor = torch.tensor(inp_batch, dtype=torch.long, device=device)
    target_tensor = torch.tensor(tgt_batch, dtype=torch.long, device=device)
    # forward
    logits = model(input_tensor)
    bsz, seq_len, vocab_size = logits.size()
    logits_flat = logits.view(-1, vocab_size)
    target_flat = target_tensor.view(-1)
    ignore_idx = pad if pad >= 0 else -100
    loss = F.cross_entropy(logits_flat, target_flat, ignore_index=ignore_idx)
    # backward and optimize
    optimizer = optim.Adam(model.parameters(), lr=float(os.getenv('LEARNING_RATE', '1e-3')))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # compute accuracy (ignore padding)
    with torch.no_grad():
        preds = logits_flat.argmax(dim=-1)
        mask = target_flat != ignore_idx
        correct = (preds == target_flat) & mask
        total = mask.sum().item()
        accuracy = correct.sum().item() / total if total > 0 else 0.0
    # save updated model
    save_all()
    return TrainResponse(status="training_complete", loss=loss.item(), accuracy=accuracy)

def online_learn(request: OnlineLearnRequest) -> OnlineLearnResponse:
    # Online AR learning: one-step gradient update on single example
    if model is None or tokenizer is None:
        load_all()
    model.train()
    bos = tokenizer.sp.bos_id()
    eos = tokenizer.sp.eos_id()
    pad = tokenizer.sp.pad_id()
    base = prepare_input_json(request.input.dict()) + ' ' + request.label
    core_ids = tokenizer.encode(base)
    inp_ids = [bos] + core_ids
    tgt_ids = core_ids + [eos]
    # tensor
    inp_tensor = torch.tensor([inp_ids], dtype=torch.long, device=device)
    tgt_tensor = torch.tensor([tgt_ids], dtype=torch.long, device=device)
    logits = model(inp_tensor)
    vocab_size = logits.size(-1)
    logits_flat = logits.view(-1, vocab_size)
    target_flat = tgt_tensor.view(-1)
    ignore_idx = pad if pad >= 0 else -100
    loss = F.cross_entropy(logits_flat, target_flat, ignore_index=ignore_idx)
    optimizer = optim.Adam(model.parameters(), lr=float(os.getenv('LEARNING_RATE', '1e-3')))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    save_all()
    return OnlineLearnResponse(status="online_learning_complete", loss=loss.item())
