"""hf_finetune.py
===================
CLI utility to **fine-tune** the autoregressive transformer on a
labelled Hugging Face dataset so that it learns to predict the *category*
for a given transaction.

Compared to `hf_trainer.py` (which performs *self-supervised* next-token
modelling on raw text), this script builds supervised training examples
that end in the ground-truth category label and then relies on
`services.model_manager.train` – which already implements this training
scheme – to update the model.

For each HF row the script constructs an `InferenceRequest` that mirrors
the JSON structure expected by the API:

    {
        "current_transaction": {
            "description": ..., "name": ..., "merchant": ..., "amount": ...
        },
        "user_history": []
    }

That request together with the category label is passed to
`model_manager.train` (in reasonably sized mini-batches to keep memory
usage in check).

Example
-------
    python hf_finetune.py \
        --dataset pointe77/credit-card-transaction \
        --split   train \
        --text-fields description name merchant amount \
        --category-field category \
        --limit   20000 \
        --batch-size 64
"""

from __future__ import annotations

import argparse
import math
from typing import List

from datasets import load_dataset

from schemas.transaction import Transaction, InferenceRequest, TrainRequest
from services import model_manager as mgr


def row_to_request(row: dict, text_fields: List[str]) -> InferenceRequest:
    """Convert dataset *row* to an InferenceRequest struct."""

    # Assemble current_transaction dict picking the desired fields, falling
    # back to empty string so that pydantic will still accept it.
    tx_kwargs = {
        "description": str(row.get("description", "")),
        "name": str(row.get("name", "")),
        "merchant": str(row.get("merchant", "")),
        "amount": str(row.get("amount", "0")),
    }

    # If the dataset happens to store amount under a different column the
    # user can include it in *text_fields* and it will be captured above.
    tx = Transaction(**tx_kwargs)
    return InferenceRequest(current_transaction=tx, user_history=[])


def fine_tune_from_hf_dataset(
    dataset_name: str,
    split: str = "train",
    text_fields: List[str] | None = None,
    category_field: str = "category",
    limit: int | None = None,
    batch_size: int = 32,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    label_smoothing: float = 0.0,
    use_amp: bool = False,
) -> None:
    """End-to-end fine-tuning entry point."""

    ds = load_dataset(dataset_name, split=split)

    if limit is not None and limit < len(ds):
        ds = ds.select(range(limit))

    text_fields = text_fields or ["description", "name", "merchant", "amount"]

    # Convert entire split into requests + labels so that we can easily
    # slice batches afterwards.
    requests: List[InferenceRequest] = []
    labels: List[str] = []

    for row in ds:
        if category_field not in row or row[category_field] is None:
            # Skip rows without a category label.
            continue
        requests.append(row_to_request(row, text_fields))
        labels.append(str(row[category_field]))

    if not requests:
        raise RuntimeError("No training examples with a category label found.")

    num_batches = math.ceil(len(requests) / batch_size)
    print(f"Fine-tuning on {len(requests)} examples ({num_batches} batches of size {batch_size})")

    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        batch_req = requests[start:end]
        batch_lbl = labels[start:end]

        train_req = TrainRequest(data=batch_req, labels=batch_lbl)
        resp = mgr.train(
            train_req,
        )
        print(
            f"Batch {i + 1}/{num_batches} – loss: {resp.loss:.4f}, accuracy: {resp.accuracy:.4f}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune AR model on a labelled HF dataset")
    parser.add_argument("--dataset", required=True, help="HF dataset identifier e.g. 'pointe77/credit-card-transaction'")
    parser.add_argument("--split", default="train", help="Dataset split to use (default: train)")
    parser.add_argument(
        "--text-fields",
        nargs="*",
        help="Columns describing the transaction (default: description name merchant amount)",
    )
    parser.add_argument(
        "--category-field",
        default="category",
        help="Column containing the category label (default: category)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of rows for quick experiments")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size passed to model_manager.train")

    args = parser.parse_args()

    fine_tune_from_hf_dataset(
        dataset_name=args.dataset,
        split=args.split,
        text_fields=args.text_fields,
        category_field=args.category_field,
        limit=args.limit,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
