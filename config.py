import os

class Config:
    MODEL_DIR = os.getenv("MODEL_DIR", "./models")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    USE_CUDA = os.getenv("USE_CUDA", "1") == "1"
    MODEL_VERSION = int(os.getenv("MODEL_VERSION", "1"))
    MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "32"))
    # SentencePiece settings
    SP_MODEL_PREFIX = os.getenv("SP_MODEL_PREFIX", "./spm")
    # Larger vocabulary improves representation capacity; 500 was too small
    # for diverse merchant/description text. Bump default to 4000 while still
    # lightweight. Override via env SP_VOCAB_SIZE if needed.
    SP_VOCAB_SIZE = int(os.getenv("SP_VOCAB_SIZE", "4000"))
    SP_TRAIN_DATA = os.getenv("SP_TRAIN_DATA", "../data/output.txt")  # CSV file path with 'text' column
    # Autoregressive generation settings
    AR_MAX_GENERATE_LENGTH = int(os.getenv("AR_MAX_GENERATE_LENGTH", "50"))
    # Enable torch.compile (PyTorch 2.x+) for runtime optimisation
    # Disabled by default so unit-tests relying on isinstance checks still pass;
    # enable via env `TORCH_COMPILE=1` in production.
    TORCH_COMPILE = os.getenv("TORCH_COMPILE", "1") == "1"
