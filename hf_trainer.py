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
import os
import tempfile
from pathlib import Path
from typing import List

from datasets import load_dataset

from config import Config
from services.model_manager import ar_train


def prepare_text_lines(dataset, fields: List[str] | None = None) -> List[str]:
    """Concatenate *fields* of each HF row into one string.

    If *fields* is *None*, all columns are used in the order returned by the
    dataset. If a field is missing for a given row the value is skipped.
    """

    lines: List[str] = []
    for row in dataset:
        if fields is None:
            values = [str(v) for v in row.values()]
        else:
            values = [str(row.get(f, "")) for f in fields]
        # drop empty strings so that we do not create double spaces
        lines.append(" ".join(v for v in values if v))
    return lines


def train_from_hf_dataset(
    dataset_name: str,
    split: str = "train",
    fields: List[str] | None = None,
    limit: int | None = None,
    epochs: int = 1,
    batch_size: int = 8,
    lr: float = 1e-3,
) -> List[float]:
    """End-to-end training util used by the CLI below."""

    ds = load_dataset(dataset_name, split=split)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    # Create temp directory that will live for the duration of the run
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        sp_text_file = tmpdir_path / "sp_text.txt"

        # Build text lines & dump into file
        lines = prepare_text_lines(ds, fields)
        sp_text_file.write_text("\n".join(lines), encoding="utf-8")

        # Point config to this temporary file / prefix so that SPTokenizer will
        # be trained on the very same text.
        Config.SP_TRAIN_DATA = str(sp_text_file)
        Config.SP_MODEL_PREFIX = str(tmpdir_path / "spm_model")

        # AR training – can reuse *lines* list we already built.
        losses = ar_train(lines, epochs=epochs, batch_size=batch_size, lr=lr)

    return losses


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AR model from HF dataset")
    parser.add_argument("--dataset", required=True, help="HF dataset identifier e.g. 'pointe77/credit-card-transaction'")
    parser.add_argument("--split", default="train", help="Dataset split to use (default: train)")
    parser.add_argument("--fields", nargs="*", help="Space-separated list of columns to concatenate. If omitted, all fields are used.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of rows for quick experiments")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)

    args = parser.parse_args()

    losses = train_from_hf_dataset(
        dataset_name=args.dataset,
        split=args.split,
        fields=args.fields,
        limit=args.limit,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    for idx, loss in enumerate(losses, start=1):
        print(f"Epoch {idx}/{len(losses)} - avg loss: {loss:.4f}")


if __name__ == "__main__":
    main()
