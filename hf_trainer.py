"""hf_trainer.py
Utility to train the autoregressive transformer **and** the SentencePiece tokenizer
directly from a dataset hosted on the Hugging Face Hub.

The script downloads the requested split, optionally limits the number of rows,
builds a temporary newline-delimited text file from the chosen columns so that
`SPTokenizer.train` sees *exactly* the same data the AR model will later see,
then invokes `services.model_manager.ar_train` to perform next-token
prediction training.

Example
-------
    python hf_trainer.py \
        --dataset pointe77/credit-card-transaction \
        --split   train \
        --fields  merchant category amt \
        --limit   10000 \
        --epochs  3 \
        --batch-size 32
"""


from __future__ import annotations

import argparse
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import List

from datasets import load_dataset

from config import Config
from services.model_manager import ar_train

# ---------------------------------------------------------------------------
# Logging setup – respect Config.LOG_LEVEL / env LOG_LEVEL.
# ---------------------------------------------------------------------------

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO),
)


def prepare_text_lines(dataset, fields: List[str] | None = None) -> List[str]:
    """Vectorised text-line preparation using *datasets.map*.

    Using the HF datasets `map` method allows parallel preprocessing with
    **pyarrow** memory-efficient pipelines and avoids Python-level iteration.
    Returns a plain list of concatenated text lines, identical to the previous
    implementation.
    """

    import os

    if fields is None:
        # When no explicit fields are given we use *all* columns.
        fields = list(dataset.column_names)

    def _concat(batch):
        out = []
        cols = [batch.get(f, []) for f in fields]
        # zip over rows
        for cells in zip(*cols):
            # skip empty strings to avoid double spaces
            out.append(" ".join(str(c) for c in cells if c))
        return {"text": out}

    num_proc = max(1, min(os.cpu_count() or 1, 16))
    mapped = dataset.map(
        _concat,
        batched=True,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        desc=f"Concatenating text fields with {num_proc} workers",
    )

    return mapped["text"]


def train_from_hf_dataset(
    dataset_name: str,
    split: str = "train",
    fields: List[str] | None = None,
    limit: int | None = None,
    epochs: int = 1,
    batch_size: int = 8,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    label_smoothing: float = 0.0,
    use_amp: bool = False,
) -> List[float]:
    """End-to-end training util used by the CLI below."""

    logger = logging.getLogger("hf_trainer")
    t0 = time.perf_counter()

    logger.info("Downloading/loading dataset '%s' split '%s'", dataset_name, split)
    ds = load_dataset(dataset_name, split=split)
    logger.info("Dataset ready in %.2fs (%d rows)", time.perf_counter() - t0, len(ds))
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    # Create temp directory that will live for the duration of the run
    # Either reuse provided SentencePiece model or create a temporary new one.
    tmp_ctx = tempfile.TemporaryDirectory() if not Config.SP_MODEL_PREFIX else None
    tmpdir_path = Path(tmp_ctx.name) if tmp_ctx else None

    # Build list of text lines once.
    prep_start = time.perf_counter()
    lines = prepare_text_lines(ds, fields)
    logger.info("Prepared %d text lines in %.2fs", len(lines), time.perf_counter() - prep_start)

    if not Config.SP_MODEL_PREFIX:  # will train a new tokenizer
        sp_text_file = tmpdir_path / "sp_text.txt"
        sp_text_file.write_text("\n".join(lines), encoding="utf-8")
        Config.SP_TRAIN_DATA = str(sp_text_file)
        Config.SP_MODEL_PREFIX = str(tmpdir_path / "spm_model")

    # AR training – reuse *lines* list.
    train_start = time.perf_counter()
    losses = ar_train(
        lines,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        label_smoothing=label_smoothing,
        use_amp=use_amp,
    )

    if tmp_ctx:
        tmp_ctx.cleanup()
    return losses


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AR model from HF dataset")
    parser.add_argument("--dataset", required=True, help="HF dataset identifier e.g. 'pointe77/credit-card-transaction'")
    parser.add_argument("--split", default="train", help="Dataset split to use (default: train)")
    parser.add_argument("--fields", nargs="*", help="Space-separated list of columns to concatenate. If omitted, all fields are used.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of rows for quick experiments")
    parser.add_argument("--sp-model", type=str, default=None,
                        help="Path to an existing SentencePiece .model to reuse (skip tokenizer training)")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable mixed-precision training (CUDA only)",
    )

    args = parser.parse_args()

    logger = logging.getLogger("hf_trainer")

    # If user supplies --sp-model we bypass tokenizer training.
    if args.sp_model:
        sp_path = Path(args.sp_model)
        if not sp_path.exists():
            raise FileNotFoundError(f"Provided --sp-model '{sp_path}' does not exist")
        Config.SP_MODEL_PREFIX = str(sp_path.with_suffix(''))
        Config.SP_TRAIN_DATA = ""  # disable automatic training in load_all

    train_start = time.perf_counter()

    losses = train_from_hf_dataset(
        dataset_name=args.dataset,
        split=args.split,
        fields=args.fields,
        limit=args.limit,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        label_smoothing=args.label_smoothing,
        use_amp=args.amp,
    )

    logger.info("Training finished in %.2fs", time.perf_counter() - train_start)

    for idx, loss in enumerate(losses, start=1):
        print(f"Epoch {idx}/{len(losses)} - avg loss: {loss:.4f}")


if __name__ == "__main__":
    main()
