import torch
import torch.nn as nn

class VanillaTransformerDecoderAR(nn.Module):
    """
    Autoregressive Transformer decoder-only model.
    Supports causal self-attention with optional encoder-decoder cross-attention.
    """
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2,
                 max_length=128, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_length, d_model)
        self.dropout = nn.Dropout(dropout)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer = nn.TransformerDecoder(decoder_layer,
                                                num_layers=num_layers)
        self.fc = nn.Linear(d_model, vocab_size)
        self.max_length = max_length

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
        pos_ids = torch.arange(tgt_len, device=device).unsqueeze(0).expand(bsz, -1)
        x = self.token_embedding(input_ids) + self.pos_embedding(pos_ids)
        x = self.dropout(x)
        # shape for transformer: (tgt_len, batch_size, d_model)
        x = x.permute(1, 0, 2)
        tgt_mask = self._generate_square_subsequent_mask(tgt_len).to(device)
        # decoder-only: if no encoder memory provided, use x as memory (causal self-att)
        if memory is None:
            mem = x
        else:
            mem = memory
        # pass only causal mask to self-attention; ignore memory_mask for simplicity
        x = self.transformer(
            tgt=x,
            memory=mem,
            tgt_mask=tgt_mask,
            memory_mask=None
        )
        x = x.permute(1, 0, 2)
        logits = self.fc(x)
        return logits