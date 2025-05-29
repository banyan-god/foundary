import os
import shutil
import torch
import pytest

import services.model_manager as mgr
from config import Config
from schemas.transaction import (
    Transaction, InferenceRequest, TrainRequest,
    BatchInferenceRequest, OnlineLearnRequest
)
from models.transformer_ar import VanillaTransformerDecoderAR


class DummySPTokenizer:
    """Fake tokenizer: minimal interface for model_manager tests."""
    def __init__(self, model_file: str):
        # pretend vocab size
        self.sp = type('S', (), {})()
        self.sp.get_piece_size = lambda: 16
        self.sp.bos_id = lambda: 1
        self.sp.eos_id = lambda: 2
        self.sp.pad_id = lambda: 0
        self.model_file = model_file
        self.vocab_file = model_file.replace('.model', '.vocab')

    def encode(self, text: str):
        # simple char-based encoding
        return [ord(c) % self.sp.get_piece_size() for c in text]

    def save(self, path_prefix: str):
        open(f"{path_prefix}.model", 'w').close()
        open(f"{path_prefix}.vocab", 'w').close()


@pytest.fixture(autouse=True)
def clean_env(tmp_path, monkeypatch):
    # redirect MODEL_DIR
    tmp_models = tmp_path / "models"
    if tmp_models.exists():
        shutil.rmtree(tmp_models)
    monkeypatch.setattr(Config, 'MODEL_DIR', str(tmp_models))
    # create dummy SP files
    prefix = tmp_path / "spm_dummy"
    mf = prefix.with_suffix('.model')
    vf = prefix.with_suffix('.vocab')
    mf.write_text('')
    vf.write_text('')
    monkeypatch.setattr(Config, 'SP_MODEL_PREFIX', str(prefix))
    # override SPTokenizer in model_manager
    monkeypatch.setattr(mgr, 'SPTokenizer', DummySPTokenizer)
    # reset manager
    mgr.model = None
    mgr.tokenizer = None
    mgr._example_map.clear()
    yield
    # cleanup
    if tmp_models.exists():
        shutil.rmtree(tmp_models)


def test_load_and_save_all_creates_model_and_files():
    # load manager: should create model & tokenizer
    mgr.load_all()
    assert isinstance(mgr.model, VanillaTransformerDecoderAR)
    assert isinstance(mgr.tokenizer, DummySPTokenizer)
    # save should write to MODEL_DIR/model_v1.pt
    mgr.save_all()
    path = mgr.get_model_path(1)
    assert os.path.isfile(path)

def test_predict_without_mapping_returns_default():
    # with no mappings, predict returns default
    tx = Transaction(description='a', name='b', merchant='c', amount='1.2')
    req = InferenceRequest(current_transaction=tx, user_history=[])
    mgr._example_map.clear()
    resp = mgr.predict(req)
    assert resp.predicted_category == ''
    assert resp.confidence == pytest.approx(0.0)

def test_predict_with_mapping_returns_mapped():
    tx = Transaction(description='x', name='y', merchant='z', amount='3.4')
    req = InferenceRequest(current_transaction=tx, user_history=[])
    key = tuple(sorted(tx.model_dump().items()))
    mgr._example_map[key] = 'HELLO'
    resp = mgr.predict(req)
    assert resp.predicted_category == 'HELLO'
    assert resp.confidence == pytest.approx(1.0)

def test_batch_predict_wraps_predict():
    # create two inference requests
    tx1 = Transaction(description='a', name='n', merchant='m', amount='5.0')
    tx2 = Transaction(description='b', name='n2', merchant='m2', amount='6.0')
    reqs = [InferenceRequest(current_transaction=tx1, user_history=[]),
            InferenceRequest(current_transaction=tx2, user_history=[])]
    batch = BatchInferenceRequest(requests=reqs)
    # monkey-patch predict to mark calls and return valid responses
    from schemas.transaction import InferenceResponse
    called = []
    def fake_predict(r):
        called.append(r)
        return InferenceResponse(predicted_category='X', confidence=0.5)
    # swap predict
    orig = mgr.predict
    mgr.predict = fake_predict
    try:
        resp = mgr.batch_predict(batch)
    finally:
        mgr.predict = orig
    # should have produced two identical InferenceResponse entries
    assert len(resp.results) == 2
    for out in resp.results:
        assert isinstance(out, InferenceResponse)
        assert out.predicted_category == 'X'
        assert out.confidence == 0.5
    # ensure our fake_predict was called with each request
    assert called == reqs

def test_train_updates_map_and_metrics_and_saves_model():
    # train on one example
    tx = Transaction(description='t', name='u', merchant='v', amount='7.0')
    req = InferenceRequest(current_transaction=tx, user_history=[])
    train_req = TrainRequest(data=[req], labels=['LBL'])
    resp = mgr.train(train_req)
    # mapping updated
    key = tuple(sorted(tx.model_dump().items()))
    assert mgr._example_map[key] == 'LBL'
    # metrics
    assert resp.status == 'training_complete'
    assert resp.loss >= 0.0
    assert 0.0 <= resp.accuracy <= 1.0
    # model saved
    assert os.path.isfile(mgr.get_model_path(1))

def test_online_learn_returns_loss_and_status():
    tx = Transaction(description='o', name='p', merchant='q', amount='8.0')
    req = InferenceRequest(current_transaction=tx, user_history=[])
    online_req = OnlineLearnRequest(input=req, label='OLBL')
    resp = mgr.online_learn(online_req)
    assert resp.status == 'online_learning_complete'
    assert resp.loss >= 0.0