import csv
import tempfile
import pytest
from fastapi.testclient import TestClient
from config import Config

@pytest.fixture
def client(tmp_path, monkeypatch):
    # Prepare a minimal CSV for SentencePiece training without pandas
    sp_csv = tmp_path / "sp_test.csv"
    with open(sp_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['text'])
        writer.writerow(['Starbucks coffee at Starbucks'])
        writer.writerow(['Uber ride across town'])
    # Override SP settings
    monkeypatch.setattr(Config, 'SP_TRAIN_DATA', str(sp_csv))
    monkeypatch.setattr(Config, 'SP_MODEL_PREFIX', str(tmp_path / "spm_test"))
    from api.main import app
    return TestClient(app)

def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "ok"
    assert "model_version" in data

def test_predict(client):
    payload = {
        "current_transaction": {
            "description": "Starbucks coffee",
            "name": "SBX",
            "merchant": "Starbucks",
            "amount": "4.50"
        },
        "user_history": [
            {
                "description": "Walmart groceries",
                "name": "WMT",
                "merchant": "Walmart",
                "amount": "50.00",
                "category": "groceries"
            }
        ]
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    result = response.json()
    assert "predicted_category" in result
    assert "confidence" in result

def test_batch_predict(client):
    payload = {
        "requests": [
            {
                "current_transaction": {
                    "description": "Starbucks coffee",
                    "name": "SBX",
                    "merchant": "Starbucks",
                    "amount": "4.50"
                },
                "user_history": []
            },
            {
                "current_transaction": {
                    "description": "Uber ride",
                    "name": "UBR",
                    "merchant": "Uber",
                    "amount": "15.00"
                },
                "user_history": []
            }
        ]
    }
    response = client.post("/batch_predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert len(data["results"]) == 2

def test_train(client):
    payload = {
        "data": [
            {
                "current_transaction": {
                    "description": "Starbucks coffee",
                    "name": "SBX",
                    "merchant": "Starbucks",
                    "amount": "4.50"
                },
                "user_history": []
            }
        ],
        "labels": ["coffee"]
    }
    response = client.post("/train", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "training_complete"
    assert "loss" in data
    assert "accuracy" in data

def test_online_learn(client):
    payload = {
        "input": {
            "current_transaction": {
                "description": "Starbucks coffee",
                "name": "SBX",
                "merchant": "Starbucks",
                "amount": "4.50"
            },
            "user_history": []
        },
        "label": "coffee"
    }
    response = client.post("/online_learn", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "online_learning_complete"
    assert "loss" in data

def test_invalid_amount(client):
    payload = {
        "current_transaction": {
            "description": "Invalid amount",
            "name": "INV",
            "merchant": "Invalid",
            "amount": "not_a_number"
        },
        "user_history": []
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_batch_size_limit(client):
    payload = {
        "requests": [
            {
                "current_transaction": {
                    "description": "Test",
                    "name": "TST",
                    "merchant": "Test",
                    "amount": "1.00"
                },
                "user_history": []
            }
        ] * 100
    }
    response = client.post("/batch_predict", json=payload)
    assert response.status_code == 400