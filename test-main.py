from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

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
    assert "predicted_category" in response.json()
    assert "confidence" in response.json()

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
    assert "results" in response.json()
    assert len(response.json()["results"]) == 2

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
    assert response.json()["status"] == "training_complete"
    assert "loss" in response.json()
    assert "accuracy" in response.json()

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
    assert response.json()["status"] == "online_learning_complete"
    assert "loss" in response.json()

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
    assert response.status_code == 422  # Pydantic validation error

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
        ] * 100  # Exceeds default MAX_BATCH_SIZE=32
    }
    response = client.post("/batch_predict", json=payload)
    assert response.status_code == 400
