#!/usr/bin/env python3
# Ensure dependencies are available
import sys
try:
    import sentencepiece  # required by SPTokenizer
except ModuleNotFoundError:
    sys.stderr.write(
        "Error: 'sentencepiece' module not found.\n"
        "Please install dependencies via 'pip install -r requirements.txt'"
        " or activate the project virtualenv.\n"
    )
    sys.exit(1)
"""
Async training script for autoregressive model.
Reads raw text data (plain lines or CSV with 'text' column),
runs self-supervised AR training, and writes epoch losses to a JSON file.
"""
import argparse
import asyncio
import csv
import json
import os

from config import Config
from services.model_manager import ar_train


def load_texts(input_path: str) -> list[str]:
    # Try CSV with 'text' column
    texts: list[str] = []
    try:
        with open(input_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            if 'text' in (reader.fieldnames or []):
                for row in reader:
                    txt = row.get('text', '').strip()
                    if txt:
                        texts.append(txt)
                return texts
    except Exception:
        pass
    # Fallback: plain text lines
    with open(input_path, encoding='utf-8') as f:
        for line in f:
            t = line.strip()
            if t:
                texts.append(t)
    return texts

async def async_train(texts: list[str], epochs: int, batch_size: int, lr: float) -> list[float]:
    loop = asyncio.get_event_loop()
    # Run blocking ar_train in executor
    return await loop.run_in_executor(None, lambda: ar_train(texts, epochs=epochs, batch_size=batch_size, lr=lr))

def main():
    parser = argparse.ArgumentParser(description="Async AR training script")
    parser.add_argument('--input', '-i', default=Config.SP_TRAIN_DATA,
                        help='Path to training data (CSV or text file)')
    parser.add_argument('--output', '-o', default='training_results.json',
                        help='Output JSON file for epoch losses')
    parser.add_argument('--epochs', '-e', type=int,
                        default=int(os.getenv('AR_EPOCHS', '1')),
                        help='Number of epochs')
    parser.add_argument('--batch-size', '-b', type=int,
                        default=int(os.getenv('AR_BATCH_SIZE', '8')),
                        help='Batch size')
    parser.add_argument('--lr', type=float,
                        default=float(os.getenv('AR_LR', '1e-3')),
                        help='Learning rate')
    args = parser.parse_args()

    if not args.input or not os.path.exists(args.input):
        print(f"Error: input file '{args.input}' not found.")
        return
    texts = load_texts(args.input)
    if not texts:
        print(f"No valid text data found in '{args.input}'")
        return
    print(f"Starting async training on {len(texts)} sequences: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}", flush=True)
    epoch_losses = asyncio.run(async_train(texts, args.epochs, args.batch_size, args.lr))
    # Write results
    result = {'epochs': args.epochs, 'batch_size': args.batch_size, 'lr': args.lr,
              'loss_per_epoch': epoch_losses}
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(f"Training complete. Results written to {args.output}", flush=True)

if __name__ == '__main__':
    main()