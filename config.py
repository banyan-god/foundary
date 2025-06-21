import os

class Config:
    MODEL_DIR = os.getenv("MODEL_DIR", "./models")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    USE_CUDA = os.getenv("USE_CUDA", "1") == "1"
    # GPU index selection (when multiple GPUs present)
    CUDA_DEVICE_ID = int(os.getenv("CUDA_DEVICE_ID", "0"))
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

    # Transformer architecture defaults (can be overridden via env)
    # Updated, more robust architecture defaults that still fit easily in CPU
    # memory for unit-tests, while giving the model greater expressive power
    # out-of-the-box.
    MODEL_D_MODEL = int(os.getenv("MODEL_D_MODEL", "256"))
    MODEL_N_HEAD = int(os.getenv("MODEL_N_HEAD", "8"))
    MODEL_NUM_LAYERS = int(os.getenv("MODEL_NUM_LAYERS", "8"))
    MODEL_MAX_LENGTH = int(os.getenv("MODEL_MAX_LENGTH", "256"))
    MODEL_DROPOUT = float(os.getenv("MODEL_DROPOUT", "0.1"))

    # Use bitsandbytes 8-bit AdamW to save memory (set env USE_8BIT_OPT=1).
    USE_8BIT_OPT = os.getenv("USE_8BIT_OPT", "1") == "1"
    # Enable torch.compile (PyTorch 2.x+) for runtime optimisation
    # Disabled by default so unit-tests relying on isinstance checks still pass;
    # enable via env `TORCH_COMPILE=1` in production.
    TORCH_COMPILE = os.getenv("TORCH_COMPILE", "1") == "1"
