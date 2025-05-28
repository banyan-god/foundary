import os

class Config:
    MODEL_DIR = os.getenv("MODEL_DIR", "./models")
    TOKENIZER_PATH = os.getenv("TOKENIZER_PATH", "./tokenizer.json")
    LABELS_PATH = os.getenv("LABELS_PATH", "./labels.json")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    USE_CUDA = os.getenv("USE_CUDA", "1") == "1"
    MODEL_VERSION = int(os.getenv("MODEL_VERSION", "1"))
    API_KEY = os.getenv("API_KEY", None)
    MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "32"))
    REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))  # seconds
