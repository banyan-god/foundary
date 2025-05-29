import os

class Config:
    MODEL_DIR = os.getenv("MODEL_DIR", "./models")
    LABELS_PATH = os.getenv("LABELS_PATH", "./labels.json")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    USE_CUDA = os.getenv("USE_CUDA", "1") == "1"
    MODEL_VERSION = int(os.getenv("MODEL_VERSION", "1"))
    MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "32"))
    # SentencePiece settings
    SP_MODEL_PREFIX = os.getenv("SP_MODEL_PREFIX", "./spm")
    SP_VOCAB_SIZE = int(os.getenv("SP_VOCAB_SIZE", "500"))
    SP_TRAIN_DATA = os.getenv("SP_TRAIN_DATA", "output.txt")  # CSV file path with 'text' column
    # Autoregressive generation settings
    AR_MAX_GENERATE_LENGTH = int(os.getenv("AR_MAX_GENERATE_LENGTH", "50"))
    # Toggle autoregressive mode
    USE_AR = os.getenv("USE_AR", "0") == "1"
