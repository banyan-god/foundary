"""
Utility script for autoregressive model training from a CSV file.

Each CSV row is concatenated into a single text sequence. The model learns to predict
the next token in sequence (self-supervised); no explicit labels needed.
"""
import argparse
import pandas as pd
from services.model_manager import ar_train


def train_from_csv(csv_path: str, epochs: int, batch_size: int, lr: float):
    """
    Read CSV, concatenate each row's fields into text, and invoke AR training.
    Returns per-epoch losses.
    """
    df = pd.read_csv(csv_path)
    # Build list of text sequences by concatenating all field values per row
    texts = []
    for _, row in df.iterrows():
        # convert each field to string and join with spaces
        texts.append(" ".join(str(v) for v in row.values))
    losses = ar_train(texts, epochs=epochs, batch_size=batch_size, lr=lr)
    for idx, loss in enumerate(losses, start=1):
        print(f"Epoch {idx}/{len(losses)} - avg loss: {loss:.4f}")
    return losses


def main():
    parser = argparse.ArgumentParser(description="Train AR model from CSV file.")
    parser.add_argument('csv_file', help='Path to input CSV file')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    args = parser.parse_args()
    train_from_csv(args.csv_file, args.epochs, args.batch_size, args.lr)


if __name__ == '__main__':
    main()