import torch
import pytest
from models.transformer_ar import VanillaTransformerDecoderAR


def test_generate_square_subsequent_mask():
    # test that mask[i, j] is True for j > i (future tokens) and False otherwise
    model = VanillaTransformerDecoderAR(vocab_size=10)
    sz = 5
    mask = model._generate_square_subsequent_mask(sz)
    assert mask.shape == (sz, sz)
    for i in range(sz):
        for j in range(sz):
            if j <= i:
                assert not mask[i, j]
            else:
                assert mask[i, j]


def test_forward_no_memory_shapes():
    # forward without encoder memory: output logits shape should match (batch, seq, vocab)
    torch.manual_seed(0)
    vocab_size = 8
    batch_size = 3
    seq_len = 4
    model = VanillaTransformerDecoderAR(vocab_size=vocab_size, d_model=16, nhead=4,
                                        num_layers=2, max_length=10)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits = model(input_ids)
    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (batch_size, seq_len, vocab_size)


def test_forward_with_memory_effect():
    # forward with different memory should produce different outputs
    torch.manual_seed(0)
    vocab_size = 8
    batch_size = 2
    seq_len = 5
    d_model = 16
    model = VanillaTransformerDecoderAR(vocab_size=vocab_size, d_model=d_model,
                                        nhead=4, num_layers=1, max_length=10)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    # random memory
    src_len = 6
    mem_rand = torch.randn(src_len, batch_size, d_model)
    mem_zero = torch.zeros_like(mem_rand)
    out_rand = model(input_ids, memory=mem_rand)
    out_zero = model(input_ids, memory=mem_zero)
    # shapes match
    assert out_rand.shape == (batch_size, seq_len, vocab_size)
    assert out_zero.shape == (batch_size, seq_len, vocab_size)
    # outputs should differ when memory differs
    assert not torch.allclose(out_rand, out_zero)