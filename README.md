# Transaction Classifier Microservice

This project implements a production-ready text classification service . It provides REST endpoints for single and batch inference, online learning, and training with persistence.

## Quick Start
1. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the server:
   ```bash
   python run.py
   ```
4. Explore the API documentation at http://localhost:8000/docs

## Usage
- Health check: `GET /health`
- Single inference: `POST /predict` with JSON payload:
  ```json
  {
    "current_transaction": { "description": "Coffee", "name": "SBX", "merchant": "Starbucks", "amount": "4.50" },
    "user_history": []
  }
  ```
- Batch inference: `POST /batch_predict`
- Train model: `POST /train`
- Online learn: `POST /online_learn`

### API Key
If `API_KEY` is set in environment, include header `x-api-key: <API_KEY>` in all requests.

## Configuration
See `config.py` for available environment variables:
- `MODEL_DIR`, `TOKENIZER_PATH`, `LABELS_PATH` — persistence locations
- `USE_CUDA`, `MAX_BATCH_SIZE`, `API_KEY`, etc.

## Developer Docs
For code structure, development, and advanced testing, see [Developer-Readme.md](Developer-Readme.md).
  
## Testing
Run all tests:
```bash
pytest -q
```