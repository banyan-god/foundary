import pytest
import shutil
from fastapi.testclient import TestClient
from config import Config
import services.model_manager as mgr
from api.main import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Redirect model artifacts to a temp dir and reset state
    tmp_models = tmp_path / "models"
    tmp_tokenizer = tmp_path / "tokenizer.json"
    monkeypatch.setattr(Config, 'MODEL_DIR', str(tmp_models))
    mgr.model = None
    mgr.tokenizer = None
    mgr.model_version = Config.MODEL_VERSION
    # Ensure clean filesystem for models and tokenizer
    for path in (tmp_models, tmp_tokenizer):
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    return TestClient(app)

def test_credit_card_transaction_classification(client):
    # Training data for four categories
    raw_train = [
        ({"description": "Walmart grocery purchase", "name": "WMT", "merchant": "Walmart", "amount": "45.00"}, "groceries"),
        ({"description": "Shell fuel refill",     "name": "SHEL", "merchant": "Shell",   "amount": "30.00"}, "gas"),
        ({"description": "Electric bill payment", "name": "ELEC", "merchant": "UtilityCo","amount": "120.00"}, "utilities"),
        ({"description": "McDonalds lunch deal", "name": "MCD",  "merchant": "McDonalds","amount": "8.50"},  "restaurant")
    ]
    # Prepare train payload
    data = []
    labels = []
    for txn, lbl in raw_train:
        data.append({"current_transaction": txn, "user_history": []})
        labels.append(lbl)
    res = client.post("/train", json={"data": data, "labels": labels})
    assert res.status_code == 200, res.text
    resp = res.json()
    assert resp.get("status") == "training_complete"
    # Test memorization: prediction on exact training samples
    for txn, lbl in raw_train:
        pl = {"current_transaction": txn, "user_history": []}
        rpt = client.post("/predict", json=pl)
        assert rpt.status_code == 200
        out = rpt.json()
        print(f"Predicted category: {out.get('predicted_category')}, expected: {lbl}")
        assert out.get("predicted_category") == lbl
        assert out.get("confidence") >= 0.0
    # Test on new but similar transactions: label should be one of known classes
    new_tests = [
        ({"description": "Trader Joe groceries", "name": "TRJ", "merchant": "TraderJoe", "amount": "60.00"}, "groceries"),
        ({"description": "BP gas station fill",   "name": "BP",  "merchant": "BP",         "amount": "25.00"}, "gas"),
        ({"description": "Water utility bill",    "name": "WTR", "merchant": "WaterCo",   "amount": "55.00"}, "utilities"),
        ({"description": "Burger King dinner",    "name": "BKR", "merchant": "BurgerKing","amount": "12.00"}, "restaurant")
    ]
    known = set(labels)
    for txn, expected in new_tests:
        rpt = client.post("/predict", json={"current_transaction": txn, "user_history": []})
        assert rpt.status_code == 200
        out = rpt.json()
        cat = out.get("predicted_category")
        print(f"Predicted category: {cat}, expected: {expected}")
        assert cat in known
        assert out.get("confidence") >= 0.0