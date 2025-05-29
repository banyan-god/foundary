# tests for the decoder-only autoregressive Transformer model
import torch
import pytest

from models.transformer_ar import VanillaTransformerDecoderAR


@pytest.mark.parametrize("vocab_size,d_model,nhead,num_layers,dropout", [
    (32, 64, 4, 2, 0.0),
    (128, 128, 8, 3, 0.1),
])
def test_decoder_output_shape(vocab_size, d_model, nhead, num_layers, dropout):
    batch_size = 4
    tgt_len = 7
    model = VanillaTransformerDecoderAR(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        max_length=20,
        dropout=dropout
    )
    # random input ids in valid range
    input_ids = torch.randint(0, vocab_size, (batch_size, tgt_len))
    # test decoder-only (no memory)
    logits = model(input_ids)
    assert logits.shape == (batch_size, tgt_len, vocab_size)

    # test with encoder memory
    src_len = 5
    # memory shape: (src_len, batch_size, d_model)
    memory = torch.randn(src_len, batch_size, d_model)
    # optional memory mask: allow all
    mem_mask = torch.zeros(src_len, src_len, dtype=torch.bool)
    logits2 = model(input_ids, memory=memory, memory_mask=mem_mask)
    assert torch.allclose(logits2, logits2)  # just ensure runs without error
    assert logits2.shape == (batch_size, tgt_len, vocab_size)

def test_causal_mask_generation():
    model = VanillaTransformerDecoderAR(vocab_size=10)
    # create mask for size 5
    mask = model._generate_square_subsequent_mask(5)
    # expected: True above diagonal, False on and below
    expected = torch.tensor([
        [False, True,  True,  True,  True ],
        [False, False, True,  True,  True ],
        [False, False, False, True,  True ],
        [False, False, False, False, True ],
        [False, False, False, False, False],
    ], dtype=torch.bool)
    assert torch.equal(mask, expected)

def test_dropout_behavior_consistency():
    # With dropout=0, model outputs should be deterministic in eval mode
    batch_size = 3
    tgt_len = 6
    vocab_size = 20
    model = VanillaTransformerDecoderAR(
        vocab_size=vocab_size,
        d_model=32,
        nhead=4,
        num_layers=1,
        max_length=10,
        dropout=0.0
    )
    model.eval()
    input_ids = torch.randint(0, vocab_size, (batch_size, tgt_len))
    out1 = model(input_ids)
    out2 = model(input_ids)
    assert torch.allclose(out1, out2, atol=1e-6)
