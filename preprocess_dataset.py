"""preprocess_dataset.py
========================
Utility script that downloads a Hugging Face dataset split, tokenises every
row with the *same* SentencePiece model that our service uses and stores the
result as an Arrow file (`.arrow`) together with a simple JSON metadata side
car.  Subsequent training runs can memory-map the prepared file directly,
skipping the costly Python tokenisation loop.

The script is intentionally lightweight so it has **no** dependency on
PyTorch; it relies only on *datasets* and the custom `SPTokenizer` wrapper
from `models.tokenizer`.

Example
-------
    python preprocess_dataset.py \
        --dataset pointe77/credit-card-transaction \
        --split   train \
        --text-fields description name merchant amount \
        --output-dir ./prepared/pointe77
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import logging, time

import os

from datasets import (
    Dataset,
    Features,
    Sequence,
    Value,
    load_dataset,
    disable_caching,
)

# Ensure we do not fill the user cache with huge tmp files.
disable_caching()

from config import Config
from models.tokenizer import SPTokenizer


def build_tokenizer() -> SPTokenizer:
    """Load existing SentencePiece or train a new one on-the-fly."""

    prefix = Path(Config.SP_MODEL_PREFIX)
    model_file = prefix.with_suffix(".model")
    if model_file.exists():
        return SPTokenizer(str(model_file))
    # Fall-back: train new tokenizer on the file passed via env
    if not Config.SP_TRAIN_DATA or not Path(Config.SP_TRAIN_DATA).exists():
        raise FileNotFoundError(
            "No SP model found and SP_TRAIN_DATA is missing. Provide a pretrained tokenizer."
        )
    SPTokenizer.train(
        input_file=Config.SP_TRAIN_DATA,
        model_prefix=str(prefix),
        vocab_size=Config.SP_VOCAB_SIZE,
    )
    return SPTokenizer(str(model_file))


def encode_batch(batch: dict, tok: SPTokenizer, fields: List[str]):
    """Vectorised encoding for *batched=True* map call.

    *batch* is a dict of column -> list values.
    Returns two lists: input_ids and their lengths.
    """

    input_ids_list = []
    lengths = []
    rows = zip(*(batch.get(f, []) for f in fields))
    for cells in rows:
        text = " ".join(str(c) for c in cells if c)
        ids = tok.encode(text)
        input_ids_list.append(ids)
        lengths.append(len(ids))
    return {"input_ids": input_ids_list, "length": lengths}


def preprocess_dataset(
    dataset_name: str,
    split: str,
    text_fields: List[str],
    output_dir: Path,
) -> None:
    logger = logging.getLogger("preprocess_dataset")
    t0 = time.perf_counter()

    logger.info("Loading dataset '%s' split '%s'", dataset_name, split)
    ds = load_dataset(dataset_name, split=split)

    logger.info("Dataset loaded in %.2fs (%d rows)", time.perf_counter() - t0, len(ds))

    tokenizer = build_tokenizer()
    logger.info("Tokenizer ready (vocab=%d)", tokenizer.sp.get_piece_size())

    features = Features({
        "input_ids": Sequence(Value("int32")),
        "length": Value("int32"),
    })

    num_proc = max(1, min(os.cpu_count() or 1, 8))  # sensible default

    map_start = time.perf_counter()
    ds_encoded: Dataset = ds.map(
        lambda batch: encode_batch(batch, tokenizer, text_fields),
        batched=True,
        remove_columns=ds.column_names,
        features=features,
        num_proc=num_proc,
        desc=f"Tokenising with {num_proc} workers",
    )

    logger.info("Tokenisation finished in %.2fs", time.perf_counter() - map_start)

    output_dir.mkdir(parents=True, exist_ok=True)
    arrow_path = output_dir / f"{dataset_name.replace('/', '_')}_{split}.arrow"
    meta_path = output_dir / "meta.json"

    save_start = time.perf_counter()
    ds_encoded.save_to_disk(str(arrow_path))
    logger.info("Saved Arrow to %s (%.2fs)", arrow_path, time.perf_counter() - save_start)

    meta = {
        "dataset": dataset_name,
        "split": split,
        "num_rows": len(ds_encoded),
        "vocab_size": tokenizer.sp.get_piece_size(),
        "fields": text_fields,
        "tokenizer_model": str(Path(Config.SP_MODEL_PREFIX).with_suffix(".model")),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Saved encoded dataset to {arrow_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-tokenise HF dataset split")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-fields", nargs="*", default=["description", "name", "merchant", "amount"],
                        help="Columns to concatenate before tokenisation")
    parser.add_argument("--output-dir", required=True, type=Path)

    args = parser.parse_args()

    preprocess_dataset(
        dataset_name=args.dataset,
        split=args.split,
        text_fields=args.text_fields,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
