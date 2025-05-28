from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "ok"
    assert "model_version" in data

def test_predict():
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

def test_batch_predict():
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

def test_train():
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

def test_online_learn():
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

def test_invalid_amount():
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

def test_batch_size_limit():
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