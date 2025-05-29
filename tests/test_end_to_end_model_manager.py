import os
import shutil
import pytest

import services.model_manager as mgr
from config import Config
from schemas.transaction import (
    Transaction, InferenceRequest, TrainRequest, BatchInferenceRequest
)

class DummySPTokenizer:
    """Fake tokenizer for end-to-end model_manager tests."""
    def __init__(self, model_file: str):
        self.sp = type('S', (), {})()
        self.sp.get_piece_size = lambda: 32
        self.sp.bos_id = lambda: 1
        self.sp.eos_id = lambda: 2
        self.sp.pad_id = lambda: 0
        self.model_file = model_file
        self.vocab_file = model_file.replace('.model', '.vocab')

    def encode(self, text: str):
        # map each char to an int in range
        return [ord(c) % self.sp.get_piece_size() for c in text]

    def save(self, path_prefix: str):
        # create dummy model/vocab files
        open(f"{path_prefix}.model", 'w').close()
        open(f"{path_prefix}.vocab", 'w').close()


@pytest.fixture(autouse=True)
def setup_env(tmp_path, monkeypatch):
    # Redirect model directory
    tmp_models = tmp_path / "models"
    if tmp_models.exists():
        shutil.rmtree(tmp_models)
    monkeypatch.setattr(Config, 'MODEL_DIR', str(tmp_models))
    # Create dummy SP model files
    prefix = tmp_path / "spm_dummy"
    mf = prefix.with_suffix('.model')
    vf = prefix.with_suffix('.vocab')
    mf.write_text('')
    vf.write_text('')
    monkeypatch.setattr(Config, 'SP_MODEL_PREFIX', str(prefix))
    # Override SPTokenizer
    monkeypatch.setattr(mgr, 'SPTokenizer', DummySPTokenizer)
    # Reset manager state
    mgr.model = None
    mgr.tokenizer = None
    mgr._example_map.clear()
    yield
    # Cleanup
    if tmp_models.exists():
        shutil.rmtree(tmp_models)


def test_end_to_end_train_and_validate():
    # Generate 100 mock transactions with 5 categories
    n = 100
    categories = ['food', 'travel', 'ent', 'supplies', 'other']
    reqs = []
    labels = []
    for i in range(n):
        cat = categories[i % len(categories)]
        tx = Transaction(
            description=f"{cat}_desc_{i}",
            name=f"{cat}_name",
            merchant=f"{cat}_merch",
            amount=str(float(i)),
        )
        reqs.append(InferenceRequest(current_transaction=tx, user_history=[]))
        labels.append(cat)
    # Train on all 100 samples
    train_req = TrainRequest(data=reqs, labels=labels)
    resp = mgr.train(train_req)
    assert resp.status == 'training_complete'
    # Mapping size equals training set
    assert len(mgr._example_map) == n
    # Validate on the same set: all predictions should match labels
    correct = 0
    for idx, r in enumerate(reqs):
        out = mgr.predict(r)
        if out.predicted_category == labels[idx]:
            correct += 1
    # Expect perfect recall on seen examples
    assert correct == n
    # Batch-validate in chunks of 30
    for i in range(0, n, 30):
        chunk = reqs[i:i+30]
        batch = BatchInferenceRequest(requests=chunk)
        batch_out = mgr.batch_predict(batch)
        assert len(batch_out.results) == len(chunk)
        for j, res in enumerate(batch_out.results):
            assert res.predicted_category == labels[i+j]
    # Report overall accuracy
    accuracy = correct / n
    assert accuracy == pytest.approx(1.0)