import torch
import torch.nn as nn

class VanillaTransformerDecoderAR(nn.Module):
    """
    Autoregressive Transformer decoder-only model.
    Supports causal self-attention with optional encoder-decoder cross-attention.
    """
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=8,
                 max_length=128, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_length, d_model)
        self.dropout = nn.Dropout(dropout)

        # Replace the built-in TransformerDecoder with our own lightweight
        # implementation so that we are independent from PyTorch's higher-level
        # Transformer stack (see `models/transformer_custom.py`).
        from .transformer_custom import (
            CustomTransformerDecoderLayer,
            CustomTransformerDecoder,
        )

        proto_layer = CustomTransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
        )
        self.transformer = CustomTransformerDecoder(proto_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, vocab_size)
        self.max_length = max_length

        # Pre-allocate a causal mask the size of *max_length*; we slice at
        # runtime so we pay the (triu) cost only once.
        full_mask = self._generate_square_subsequent_mask(max_length)
        # Boolean mask expected by TransformerDecoderLayer
        self.register_buffer("_causal_mask", full_mask, persistent=False)

    def _generate_square_subsequent_mask(self, sz: int):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask

    def forward(self,
                input_ids: torch.LongTensor,
                memory: torch.Tensor = None,
                memory_mask: torch.Tensor = None):
        """
        Args:
            input_ids: (batch_size, tgt_seq_len)
            memory: encoder outputs, shape (src_seq_len, batch_size, d_model)
            memory_mask: optional mask for encoder-decoder attention
        Returns:
            logits: (batch_size, tgt_seq_len, vocab_size)
        """
        bsz, tgt_len = input_ids.size()
        device = input_ids.device
        pos_ids = torch.arange(tgt_len, device=device)
        x = self.token_embedding(input_ids) + self.pos_embedding(pos_ids)
        x = self.dropout(x)

        # Use *batch_first* = (B, T, C) so no permute is needed.
        tgt_mask = self._causal_mask[:tgt_len, :tgt_len]
        # Decoder-only: for pure causal LM we **do not** want future-token
        # information to leak through encoder-decoder cross-attention.  Passing
        # the *same* sequence as `memory` (as the previous implementation did)
        # breaks autoregressive training because cross-attention has full
        # visibility of all positions.  Instead, when no external memory is
        # given we supply a single zero vector, effectively disabling
        # cross-attention while keeping the API compatible.
        if memory is None:
            d_model = self.token_embedding.embedding_dim
            # (B, src_len=1, d_model) for batch_first
            mem = torch.zeros(bsz, 1, d_model, device=device)
        else:
            mem = memory
            if mem.ndim == 3:
                # Convert legacy (src_len, B, d) to (B, src_len, d)
                if mem.shape[1] == bsz and mem.shape[0] != bsz:
                    mem = mem.permute(1, 0, 2).contiguous()
        # pass only causal mask to self-attention; ignore memory_mask for simplicity
        x = self.transformer(
            tgt=x, memory=mem, tgt_mask=tgt_mask, memory_mask=None
        )
        logits = self.fc(x)
        return logits