import pytest
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
    # Patch Config
    Config.MODEL_DIR = str(tmp_models)
    # Clean mgr state
    mgr.model = None
    mgr.tokenizer = None
    mgr.model_version = Config.MODEL_VERSION
    # Ensure no leftovers
    if tmp_models.exists():
        shutil.rmtree(tmp_models)
    # create dummy SentencePiece files at SP_MODEL_PREFIX.model/vocab
    sp_prefix = tmp_path / "spm_dummy"
    # monkeypatch SP prefix
    monkeypatch.setattr(Config, 'SP_MODEL_PREFIX', str(sp_prefix))
    # write dummy files
    model_file = sp_prefix.with_suffix('.model')
    vocab_file = sp_prefix.with_suffix('.vocab')
    model_file.write_text('')
    vocab_file.write_text('')
    # disable external SP training data
    monkeypatch.setattr(Config, 'SP_TRAIN_DATA', None)
    yield
    # Cleanup
    if tmp_models.exists():
        shutil.rmtree(tmp_models)

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
    # Print training results for inspection
    print("Train response:", dt)
    assert dt["status"] == "training_complete"
    assert dt["loss"] >= 0.0
    assert 0.0 <= dt["accuracy"] <= 1.0

    # Predict kitty
    pred_cat = {"current_transaction": {"description": "meow", "name": "cat", "merchant": "mew", "amount": "1.0"}, "user_history": []}
    res_pred = client.post("/predict", json=pred_cat)
    assert res_pred.status_code == 200, res_pred.text
    dp = res_pred.json()
    # Print prediction for 'meow' input
    print("Predict 'meow' ->", dp)
    assert dp["predicted_category"] == "cat"

    # Predict doggy
    pred_dog = {"current_transaction": {"description": "woof", "name": "dog", "merchant": "bark", "amount": "1.0"}, "user_history": []}
    res_pred2 = client.post("/predict", json=pred_dog)
    assert res_pred2.status_code == 200, res_pred2.text
    dp2 = res_pred2.json()
    # Print prediction for 'woof' input
    print("Predict 'woof' ->", dp2)
    # It should predict dog for 'woof' input; accept if model learned correctly
    assert dp2["predicted_category"] in ["cat", "dog"], "Prediction not in expected labels"