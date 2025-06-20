"""
Utility script for autoregressive model training from a CSV file.

Each CSV row is concatenated into a single text sequence. The model learns to predict
the next token in sequence (self-supervised); no explicit labels needed.
"""
import argparse
import pandas as pd
from services.model_manager import ar_train
# Huggingface datasets support
load_dataset = None
try:
    from datasets import load_dataset
    _HAS_DATASETS = True
except ImportError:
    _HAS_DATASETS = False


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
    
def train_from_hf(dataset_name: str, split: str, epochs: int, batch_size: int, lr: float):
    """
    Load a Huggingface dataset by name, extract text fields, and train AR model.
    """
    if not _HAS_DATASETS:
        raise ImportError("datasets library is required for Huggingface dataset support. Please install via 'pip install datasets'.")
    ds = load_dataset(dataset_name)
    # ds may be a dict of splits or a single Dataset
    if isinstance(ds, dict):
        if split not in ds:
            raise ValueError(f"Split '{split}' not found in dataset '{dataset_name}'. Available splits: {list(ds.keys())}")
        data = ds[split]
    else:
        data = ds
    texts = []
    for row in data:
        # concatenate all field values into text sequence
        texts.append(" ".join(str(v) for v in row.values()))
    losses = ar_train(texts, epochs=epochs, batch_size=batch_size, lr=lr)
    for idx, loss in enumerate(losses, start=1):
        print(f"Epoch {idx}/{len(losses)} - avg loss: {loss:.4f}")
    return losses


def main():
    parser = argparse.ArgumentParser(description="Train AR model from CSV file.")
    parser.add_argument('csv_file', help='Path to input CSV file')
    parser.add_argument('--hf-dataset', type=str, default=None,
                        help='Huggingface dataset identifier (e.g., "username/dataset")')
    parser.add_argument('--hf-split', type=str, default='train',
                        help='Dataset split to use when training from Huggingface dataset')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    args = parser.parse_args()
    if args.hf_dataset:
        train_from_hf(args.hf_dataset, args.hf_split, args.epochs, args.batch_size, args.lr)
    else:
        train_from_csv(args.csv_file, args.epochs, args.batch_size, args.lr)


if __name__ == '__main__':
    main()