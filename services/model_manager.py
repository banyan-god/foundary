import os
import csv
import logging
import threading
import torch
import torch.nn.functional as F
import torch.optim as optim
import random
# ---------------------------------------------------------------------------
# Constants / globals
# ---------------------------------------------------------------------------
# Index that the loss function will ignore.  We write this value into *target*
# padding positions so real <unk> (id 0) tokens are still trained on.
IGNORE_IDX = -100

# Re‑used optimiser so momentum & Adam moments survive across .train() calls

_optimizer = None

# Persist checkpoints only when this env‑var is set.
#   export SAVE_MODEL_ON_TRAIN=1   -> save each call
#   unset  / set to 0             -> no autosave
SAVE_ON_TRAIN = os.getenv("SAVE_MODEL_ON_TRAIN", "0") == "1"

import time
# bitsandbytes is optional; import lazily when required.
from config import Config
from models.transformer_ar import VanillaTransformerDecoderAR
from models.tokenizer import SPTokenizer
from schemas.transaction import (
    InferenceRequest, InferenceResponse,
    BatchInferenceRequest, BatchInferenceResponse,
    TrainRequest, TrainResponse,
    OnlineLearnRequest, OnlineLearnResponse,
    GenerateRequest, GenerateResponse
)

model_lock = threading.Lock()
_use_cuda = Config.USE_CUDA and torch.cuda.is_available()
_use_mps = False
try:
    _use_mps = torch.backends.mps.is_available() and torch.backends.mps.is_built()
except Exception:
    _use_mps = False

# Select explicit GPU index when multiple GPUs are present.
if _use_cuda:
    idx = Config.CUDA_DEVICE_ID
    try:
        torch.cuda.set_device(idx)
        device = torch.device(f"cuda:{idx}")
    except Exception:
        device = torch.device("cuda")
else:
    device = torch.device("mps" if _use_mps else "cpu")
model = None
model_version = Config.MODEL_VERSION
tokenizer = None
# simple in-memory mapping for classification keys to labels
_example_map = {}
logger = logging.getLogger("transaction_classifier")
logger.info(f"Using device: {device}")
# Set DEBUG to see per-step timing, INFO for epoch timing.

# Opt-in to TF32 on Ampere+ when on CUDA for faster matmul with minimal accuracy hit.
if device.type == "cuda":
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

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
    # Prepare tokenizer model: train from SP_TRAIN_DATA if available, else use shipped model
    if Config.SP_TRAIN_DATA and os.path.exists(Config.SP_TRAIN_DATA):
        prefix = Config.SP_MODEL_PREFIX
        sp_model = f"{prefix}.model"
        if not os.path.exists(sp_model):
            SPTokenizer.train(
                input_file=Config.SP_TRAIN_DATA,
                model_prefix=prefix,
                vocab_size=Config.SP_VOCAB_SIZE,
            )
    else:
        sp_model = os.path.join(os.getcwd(), 'spm.model')
    if not os.path.exists(sp_model):
        raise RuntimeError("No SentencePiece model found for tokenizer.")

    logger.info(f"Starting load_all: SP_MODEL_PREFIX={Config.SP_MODEL_PREFIX}")
    tokenizer = SPTokenizer(sp_model)
    # init or load AR model
    vocab_size = tokenizer.sp.get_piece_size()
    model = VanillaTransformerDecoderAR(
        vocab_size=vocab_size,
        d_model=Config.MODEL_D_MODEL,
        nhead=Config.MODEL_N_HEAD,
        num_layers=Config.MODEL_NUM_LAYERS,
        max_length=Config.MODEL_MAX_LENGTH,
        dropout=Config.MODEL_DROPOUT,
    ).to(device)

    # Optional torch.compile for speed (PyTorch 2.x+)
    # Compile only when explicitly enabled **and** running on CUDA to avoid
    # instabilities on some back-ends (e.g. MPS) and during the test suite.
    run_under_pytest = bool(os.getenv("PYTEST_CURRENT_TEST"))
    if (
        Config.TORCH_COMPILE
        and device.type == "cuda"
        and not run_under_pytest
    ):
        compile_fn = getattr(torch, "compile", None)
        if callable(compile_fn):
            try:
                model = compile_fn(model)
                logger.info("Model compiled with torch.compile")
            except Exception as compile_exc:
                # Fallback silently – don't break if compile fails (e.g. unsupported backend)
                logger.warning(
                    "torch.compile failed – proceeding with eager model. Error: %s",
                    compile_exc,
                )
    model_file = get_model_path(1)
    if os.path.exists(model_file):
        try:
            state = torch.load(model_file, map_location=device)
            model.load_state_dict(state)
        except Exception:
            pass
    logger.info(f"load_all completed: model={model}, tokenizer={tokenizer}")
#    return
#    return

def prepare_input_json(req_dict: dict) -> str:
    parts = []
    field_sep = " | "            # explicit separator avoids token ambiguity
    # historical transactions
    for hist in req_dict.get("user_history", []):
        parts.append(
            field_sep.join(
                [str(hist.get(k, "")) for k in ("description",
                                                "name",
                                                "merchant",
                                                "amount",
                                                "category")]
            )
        )
    # current transaction
    cur = req_dict["current_transaction"]
    parts.append(
        field_sep.join(
            [str(cur.get(k, "")) for k in ("description",
                                           "name",
                                           "merchant",
                                           "amount")]
        )
    )
    # newline separates history ↔ current transaction blocks
    return "\n".join(parts)

# ---------- Padding‑mask helper ---------------------------------
def _make_pad_mask(batch_tensor: torch.Tensor, pad_id: int) -> torch.BoolTensor:
    """
    Returns BoolTensor [B, T] where **True** marks PAD tokens that should
    be ignored by attention.
    """
    return batch_tensor.eq(pad_id)

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
        cur = request.model_dump().get('current_transaction', {})
        key = tuple(sorted(cur.items()))
        if key in _example_map:
            return InferenceResponse(predicted_category=_example_map[key], confidence=1.0)
        # fallback to first known label if any
        default = next(iter(_example_map.values()), '')
        # Only attempt generation when no default labels are available
        if default == '':
            try:
                bos = tokenizer.sp.bos_id()
                prompt = prepare_input_json(request.model_dump())
                seq_ids = [bos] + tokenizer.encode(prompt)
                max_len_cap = getattr(model, 'max_length', None)
                if max_len_cap and len(seq_ids) > max_len_cap:
                    seq_ids = seq_ids[-max_len_cap:]

                input_ids = torch.tensor([seq_ids], dtype=torch.long, device=device)
                tokens = []
                max_new = 10  # small generation window to keep latency low
                with torch.no_grad():
                    for _ in range(max_new):
                        logits = model(input_ids)
                        last = logits[0, -1, :]
                        idx = int(torch.argmax(last).item())
                        # stop if EOS generated
                        if idx == tokenizer.sp.eos_id():
                            break
                        tokens.append(idx)
                        input_ids = torch.cat(
                            [input_ids, torch.tensor([[idx]], device=device)], dim=1
                        )
                        if max_len_cap and input_ids.size(1) > max_len_cap:
                            input_ids = input_ids[:, -max_len_cap:]
                # decode generated tokens -> text (best effort)
                try:
                    gen_text = tokenizer.decode(tokens).strip()
                except Exception:
                    gen_text = ''
                # heuristic: first whitespace-separated token is predicted category
                if gen_text:
                    predicted = gen_text.split()[0]
                    if predicted:
                        return InferenceResponse(
                            predicted_category=predicted, confidence=0.5
                        )
            except Exception:
                # Any failure: silently fall back to default behaviour
                pass

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
        cur = req.model_dump().get('current_transaction', {})
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
    # SentencePiece returns -1 when pad token is not defined; on CUDA a negative
    # index will trigger a device-side assert in nn.Embedding.  Map it to 0
    # (conventionally <unk>) so that it references a valid row in the embedding
    # matrix while still being ignored via *ignore_index* during loss.
    if pad < 0:
        pad = 0
    batch_inputs = []
    batch_targets = []
    for req, lbl in zip(request.data, request.labels):
        base = prepare_input_json(req.model_dump()) + ' ' + lbl
        core_ids = tokenizer.encode(base)
        # Truncate to fit positional-embedding limit if the loaded model defines it
        max_len_cap = getattr(model, "max_length", None)
        if max_len_cap and len(core_ids) + 2 > max_len_cap:
            core_ids = core_ids[: max_len_cap - 2]
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
        tgt_batch.append(tgt_ids + [IGNORE_IDX] * pad_count)
    input_tensor = torch.tensor(inp_batch, dtype=torch.long, device=device)
    target_tensor = torch.tensor(tgt_batch, dtype=torch.long, device=device)
    # forward
    pad_mask = _make_pad_mask(input_tensor, pad)
    logits = model(input_tensor, memory=None, key_padding_mask=pad_mask)
    bsz, seq_len, vocab_size = logits.size()
    logits_flat = logits.view(-1, vocab_size)
    target_flat = target_tensor.view(-1)
    ignore_idx = IGNORE_IDX
    loss = F.cross_entropy(logits_flat, target_flat, ignore_index=ignore_idx)
    # backward and optimize
    global _optimizer
    if _optimizer is None:
        _optimizer = optim.Adam(model.parameters(),
                                lr=float(os.getenv("LEARNING_RATE", "1e-3")))
    optimizer = _optimizer
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
    # optionally persist
    if SAVE_ON_TRAIN:
        save_all()
    return TrainResponse(status="training_complete", loss=loss.item(), accuracy=accuracy)

def online_learn(request: OnlineLearnRequest) -> OnlineLearnResponse:
    # Online AR learning: one-step gradient update on single example
    global _optimizer
    if model is None or tokenizer is None:
        load_all()
    model.train()
    bos = tokenizer.sp.bos_id()
    eos = tokenizer.sp.eos_id()
    pad = tokenizer.sp.pad_id()
    if pad < 0:
        pad = 0
    base = prepare_input_json(request.input.model_dump()) + ' ' + request.label
    core_ids = tokenizer.encode(base)
    max_len_cap = getattr(model, "max_length", None)
    if max_len_cap and len(core_ids) + 2 > max_len_cap:
        core_ids = core_ids[: max_len_cap - 2]
    inp_ids = [bos] + core_ids
    tgt_ids = core_ids + [eos]
    # tensor
    inp_tensor = torch.tensor([inp_ids], dtype=torch.long, device=device)
    tgt_tensor = torch.tensor([tgt_ids], dtype=torch.long, device=device)
    pad_mask = _make_pad_mask(inp_tensor, pad)
    logits = model(inp_tensor, memory=None, key_padding_mask=pad_mask)
    vocab_size = logits.size(-1)
    logits_flat = logits.view(-1, vocab_size)
    target_flat = tgt_tensor.view(-1)
    ignore_idx = IGNORE_IDX
    loss = F.cross_entropy(logits_flat, target_flat, ignore_index=ignore_idx)
    if _optimizer is None:
        _optimizer = optim.Adam(model.parameters(),
                                lr=float(os.getenv("LEARNING_RATE", "1e-3")))
    optimizer = _optimizer
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if SAVE_ON_TRAIN:
        save_all()
    return OnlineLearnResponse(status="online_learning_complete", loss=loss.item())
    
def generate(request: GenerateRequest) -> GenerateResponse:
    """Autoregressively generate next token IDs from a prompt."""
    with model_lock:
        if model is None or tokenizer is None:
            load_all()
        model.eval()
        # build prompt string
        prompt = prepare_input_json(request.model_dump())
        bos = tokenizer.sp.bos_id()
        seq_ids = [bos] + tokenizer.encode(prompt)
        # cap initial prompt to model max_length
        max_len_cap = getattr(model, 'max_length', None)
        if max_len_cap and len(seq_ids) > max_len_cap:
            seq_ids = seq_ids[-max_len_cap:]
        input_ids = torch.tensor([seq_ids], dtype=torch.long, device=device)
        # cache PAD id so we can ban it during sampling
        pad_id = tokenizer.sp.pad_id()
        tokens = []
        max_new = request.max_new_tokens or Config.AR_MAX_GENERATE_LENGTH
        with torch.no_grad():
            for _ in range(max_new):
                logits = model(input_ids)
                last = logits[0, -1, :]
                # ── Ban PAD from being selected ─────────────────────────────
                if pad_id >= 0:                       # pad_id = 3 in default SPM
                    last[pad_id] = -float("inf")
                idx = int(torch.argmax(last).item())
                if idx == tokenizer.sp.eos_id():
                    break
                tokens.append(idx)
                # append new token
                input_ids = torch.cat([input_ids, torch.tensor([[idx]], device=device)], dim=1)
                # ensure we don't exceed model's max_length (position embeddings)
                max_len_cap = getattr(model, 'max_length', None)
                if max_len_cap and input_ids.size(1) > max_len_cap:
                    input_ids = input_ids[:, -max_len_cap:]
        # decode tokens to text if supported
        try:
            text = tokenizer.decode(tokens)
        except Exception:
            text = ''
        return GenerateResponse(tokens=tokens, text=text)
    
# --------- Training Utility -------------------------------------------------
# Extended with modern training tricks: AdamW, cosine schedule, warm-up, label
# smoothing, gradient clipping and optional mixed-precision via AMP.
# ---------------------------------------------------------------------------


def ar_train(
    texts,
    epochs: int = 1,
    batch_size: int = 8,
    lr: float = 1e-3,
    *,
    weight_decay: float = 0.01,
    betas=(0.9, 0.95),
    warmup_ratio: float = 0.1,
    max_grad_norm: float = 1.0,
    label_smoothing: float = 0.0,
    use_amp: bool = False,
):
    """
    Self-supervised AR training: predict next token on concatenated input texts.
    :param texts: List of raw string sequences.
    :param epochs: Number of training epochs.
    :param batch_size: Batch size per step.
    :param lr: Learning rate for optimizer.
    :return: List of average loss per epoch.
    """
    if model is None or tokenizer is None:
        load_all()
    model.train()
    bos = tokenizer.sp.bos_id()
    eos = tokenizer.sp.eos_id()
    pad = tokenizer.sp.pad_id()
    if pad < 0:
        pad = 0
    # ------------------------------------------------------------------
    # Build input-target pairs using HuggingFace datasets map for
    # parallel tokenisation when *texts* is large.  Falls back to a plain
    # Python loop for small lists to avoid multiprocessing overhead.
    # ------------------------------------------------------------------

    if len(texts) > 1000:
        from datasets import Dataset
        import os

        max_len_cap = getattr(model, "max_length", None)
        ds = Dataset.from_dict({"text": list(texts)})

        def _tok(batch):
            inps, tgts = [], []
            for t in batch["text"]:
                ids = tokenizer.encode(t)
                if max_len_cap and len(ids) + 2 > max_len_cap:
                    ids = ids[: max_len_cap - 2]
                inps.append([bos] + ids)
                tgts.append(ids + [eos])
            return {"inp": inps, "tgt": tgts}

        num_proc = max(1, min(os.cpu_count() or 1, 8))
        ds = ds.map(_tok, batched=True, num_proc=num_proc, desc="Tokenising")
        seqs = list(zip(ds["inp"], ds["tgt"]))
    else:
        seqs = []
        max_len_cap = getattr(model, "max_length", None)
        for text in texts:
            ids = tokenizer.encode(text)
            if max_len_cap and len(ids) + 2 > max_len_cap:
                ids = ids[: max_len_cap - 2]
            seqs.append(([bos] + ids, ids + [eos]))
    # Optimizer & scheduler
    params = list(model.parameters())
    # Choose between standard AdamW and bitsandbytes 8-bit AdamW
    if params:
        use_bnb = Config.USE_8BIT_OPT
        if use_bnb is None:              # auto‑detect
            use_bnb = (device.type == "cuda")

        if use_bnb:
            try:
                import bitsandbytes as bnb

                optimizer = bnb.optim.AdamW8bit(
                    params, lr=lr, betas=betas, weight_decay=weight_decay
                )
                logger.info("Using 8-bit AdamW optimizer from bitsandbytes")
            except Exception as exc:
                logger.warning(
                    "bitsandbytes AdamW8bit unavailable (%s). Falling back to torch AdamW.",
                    exc,
                )
                optimizer = optim.AdamW(
                    params, lr=lr, betas=betas, weight_decay=weight_decay
                )
        else:
            optimizer = optim.AdamW(
                params, lr=lr, betas=betas, weight_decay=weight_decay
            )
    else:
        optimizer = None

    # Total steps = batches per epoch * epochs
    total_steps = ((len(seqs) + batch_size - 1) // batch_size) * epochs
    if optimizer:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps
        )
    else:
        scheduler = None

    if optimizer and use_amp and device.type == "cuda":
        from torch.amp import GradScaler as _GradScaler

        scaler = _GradScaler()
    else:
        scaler = None
    epoch_losses = []
    for epoch in range(1, epochs + 1):
        random.shuffle(seqs)
        total_loss = 0.0
        steps = 0
        epoch_token_count = 0
        epoch_start = time.perf_counter()
        for i in range(0, len(seqs), batch_size):
            batch = seqs[i:i+batch_size]
            max_len = max(len(inp) for inp, _ in batch)
            inp_batch, tgt_batch = [], []
            for inp, tgt in batch:
                pad_count = max_len - len(inp)
                inp_batch.append(inp + [pad] * pad_count)
                tgt_batch.append(tgt + [IGNORE_IDX] * pad_count)
            x = torch.tensor(inp_batch, dtype=torch.long, device=device)
            y = torch.tensor(tgt_batch, dtype=torch.long, device=device)

            step_token_count = x.numel()
            step_start = time.perf_counter()

            if use_amp:
                with torch.autocast("cuda"):
                    pad_mask = _make_pad_mask(x, pad)
                    logits = model(x, memory=None, key_padding_mask=pad_mask)
            else:
                pad_mask = _make_pad_mask(x, pad)
                logits = model(x, memory=None, key_padding_mask=pad_mask)
            bsz, seq_len, vocab_size = logits.size()
            logits_flat = logits.view(-1, vocab_size)
            target_flat = y.view(-1)
            loss = F.cross_entropy(
                logits_flat,
                target_flat,
                ignore_index=IGNORE_IDX,
                label_smoothing=label_smoothing,
            )

            if optimizer:
                optimizer.zero_grad()
                if use_amp:
                    scaler.scale(loss).backward()
                    # gradient clipping after unscale
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_grad_norm
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_grad_norm
                    )
                    optimizer.step()
                if scheduler:
                    scheduler.step()

            # logging
            step_time = time.perf_counter() - step_start
            logger.debug(
                "epoch %d | batch %d/%d | loss %.4f | tokens %d | %.2f tok/s",
                epoch,
                steps + 1,
                (len(seqs) + batch_size - 1) // batch_size,
                loss.item(),
                step_token_count,
                step_token_count / max(step_time, 1e-4),
            )

            epoch_token_count += step_token_count
            total_loss += loss.item()
            steps += 1
        avg = total_loss / steps if steps else 0.0
        ppl = (torch.exp(torch.tensor(avg))).item() if avg < 20 else float("inf")
        epoch_time = time.perf_counter() - epoch_start
        tok_per_sec = epoch_token_count / max(epoch_time, 1e-4)

        logger.info(
            "Epoch %d/%d finished: loss %.4f | ppl %.2f | tokens %d | %.2f tok/s | %.2fs",
            epoch,
            epochs,
            avg,
            ppl,
            epoch_token_count,
            tok_per_sec,
            epoch_time,
        )
        epoch_losses.append(avg)
    if SAVE_ON_TRAIN:
        save_all()
    return epoch_losses
