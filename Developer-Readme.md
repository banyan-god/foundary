# Developer README

This document provides an overview of the codebase, its structure, and instructions for local development and testing.

## Project Overview
- A FastAPI microservice for text-based transaction classification.
- Supports online learning, batch inference, and model persistence.
- Uses a simple Transformer encoder built with PyTorch.
- Configurable to run on CPU, CUDA (GPU), or MPS (macOS Metal Performance Shaders).

## Repository Structure
```
├── api/                # FastAPI routers and entry point
│   ├── main.py        # App instantiation, startup/shutdown events
│   ├── health.py      # GET /health endpoint
│   ├── inference.py   # POST /predict and /batch_predict
│   ├── training.py    # POST /train and /online_learn
│   └── auth.py        # API key header dependency

├── services/           # Business logic / model management
│   └── model_manager.py # Thread-safe load/save, predict/train functions

├── models/             # Model and tokenizer definitions
│   ├── transformer.py  # VanillaTransformerClassifier
│   └── tokenizer.py    # SimpleTokenizer

├── schemas/            # Pydantic request/response models
│   └── transaction.py  # Transaction, InferenceRequest/Response, etc.

├── tests/              # pytest suites
│   ├── test_api.py     # API endpoint tests (TestClient)
│   └── test_model_training.py # End-to-end train & predict test

├── config.py           # Configuration via environment variables
├── run.py              # Uvicorn entry: `python run.py`
├── requirements.txt    # Python dependencies
├── .gitignore          # Specifies ignored files (incl. model files *.pt)
└── Developer-Readme.md # This file
```

## Configuration
- All settings are managed in `config.py`, overridable via ENV vars:
  - `MODEL_DIR`, `TOKENIZER_PATH`, `LABELS_PATH` – persistence paths
  - `USE_CUDA`, `MAX_BATCH_SIZE`, `API_KEY`, etc.

## Running Locally
1. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the service:
   ```bash
   python run.py
   ```
   Service will listen on `http://0.0.0.0:8000`.

## Testing
- Run all tests with:
  ```bash
  pytest -q
  ```
- Test coverage includes:
  - API endpoint behavior (`tests/test_api.py`)
  - End-to-end model training & predict (`tests/test_model_training.py`)

## Model Management
- `services/model_manager.py` handles:
  - Loading and saving model state (`.pt`), tokenizer, and labels JSON.
  - Thread-safe inference and training via a lock.
  - Endpoints call into these functions for core logic.

## Device Support
- Automatically selects device in priority: CUDA > MPS (macOS) > CPU.

## Next Steps
- Integrate production-grade logging and monitoring.
- Expand the tokenizer and model capacity for larger datasets.
- Add batch-training and scheduled retraining.
  
For questions or contributions, please open an issue or PR. Happy coding!