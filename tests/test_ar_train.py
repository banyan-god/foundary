import torch
import pytest

import services.model_manager as mgr


class DummyTokenizer:
    def __init__(self):
        self.sp = type('S', (), {})()
        # Dummy special IDs
        self.sp.bos_id = lambda: 0
        self.sp.eos_id = lambda: 1
        self.sp.pad_id = lambda: -100

    def encode(self, text: str):
        # Map each char to small integer
        return [ord(c) % 5 for c in text]


class DummyModel(torch.nn.Module):
    def __init__(self, vocab_size=5):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, x, memory=None):
        # Return zeros logits to force zero loss
        bsz, seq_len = x.shape
        return torch.zeros(bsz, seq_len, self.vocab_size, device=x.device)


@pytest.fixture(autouse=True)
def setup_dummy(tmp_path, monkeypatch):
    # Replace model and tokenizer with dummies
    monkeypatch.setattr(mgr, 'model', DummyModel(), raising=False)
    monkeypatch.setattr(mgr, 'tokenizer', DummyTokenizer(), raising=False)
    # Prevent saving actual files
    monkeypatch.setattr(mgr, 'save_all', lambda: None)
    yield


def test_ar_train_basic():
    texts = ["abc", "de"]
    # Two epochs, batch size 1
    losses = mgr.ar_train(texts, epochs=2, batch_size=1, lr=1e-3)
    assert isinstance(losses, list)
    assert len(losses) == 2
    # Since logits are zero and target always zero, loss should be -log(1/vocab)=constant
    # But here target_flat == 0 and logits_flat are zero; cross_entropy without softmax gives log(vocab)
    for loss in losses:
        assert isinstance(loss, float)
        assert loss >= 0.0


def test_ar_train_batching():
    # Larger batch size covers multiple examples
    texts = ["a", "b", "c", "d"]
    losses = mgr.ar_train(texts, epochs=1, batch_size=4, lr=1e-3)
    assert len(losses) == 1
    assert losses[0] >= 0.0