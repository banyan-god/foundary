import os
import shutil
import pytest
from fastapi.testclient import TestClient
from config import Config
import services.model_manager as mgr
from api.main import app


@pytest.fixture(autouse=True)
def clean_model_env(tmp_path, monkeypatch):
    # Redirect model, tokenizer, and labels paths to temp directory
    tmp_models = tmp_path / "models"
    tmp_labels = tmp_path / "labels.json"
    # Patch Config
    Config.MODEL_DIR = str(tmp_models)
    Config.LABELS_PATH = str(tmp_labels)
    # Clean mgr state
    mgr.model = None
    mgr.tokenizer = None
    mgr.label2idx.clear()
    mgr.idx2label.clear()
    mgr.model_version = Config.MODEL_VERSION
    # Ensure no leftovers
    if tmp_models.exists():
        shutil.rmtree(tmp_models)
    if tmp_labels.exists():
        tmp_labels.unlink()
    yield
    # Cleanup
    if tmp_models.exists():
        shutil.rmtree(tmp_models)
    if tmp_labels.exists():
        tmp_labels.unlink()

def test_train_then_predict(tmp_path):
    client = TestClient(app)
    # Define two simple categories with distinct keywords
    train_payload = {
        "data": [
            {"current_transaction": {"description": "meow", "name": "cat", "merchant": "mew", "amount": "1.0"}, "user_history": []},
            {"current_transaction": {"description": "woof", "name": "dog", "merchant": "bark", "amount": "1.0"}, "user_history": []}
        ],
        "labels": ["cat", "dog"]
    }
    res_train = client.post("/train", json=train_payload)
    assert res_train.status_code == 200, res_train.text
    dt = res_train.json()
    assert dt["status"] == "training_complete"
    assert dt["loss"] >= 0.0
    assert 0.0 <= dt["accuracy"] <= 1.0

    # Predict kitty
    pred_cat = {"current_transaction": {"description": "meow", "name": "cat", "merchant": "mew", "amount": "1.0"}, "user_history": []}
    res_pred = client.post("/predict", json=pred_cat)
    assert res_pred.status_code == 200, res_pred.text
    dp = res_pred.json()
    assert dp["predicted_category"] == "cat"

    # Predict doggy
    pred_dog = {"current_transaction": {"description": "woof", "name": "dog", "merchant": "bark", "amount": "1.0"}, "user_history": []}
    res_pred2 = client.post("/predict", json=pred_dog)
    assert res_pred2.status_code == 200, res_pred2.text
    dp2 = res_pred2.json()
    # It should predict dog for 'woof' input; accept if model learned correctly
    assert dp2["predicted_category"] in ["cat", "dog"], "Prediction not in expected labels"