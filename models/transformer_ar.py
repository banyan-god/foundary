import torch
import torch.nn as nn

class VanillaTransformerAR(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, max_length=128):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_length, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, vocab_size)
        self.max_length = max_length

    def _generate_square_subsequent_mask(self, sz: int):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask

    def forward(self, input_ids: torch.LongTensor):
        # input_ids: (batch_size, seq_len)
        bsz, seq_len = input_ids.size()
        device = input_ids.device
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        x = self.token_embedding(input_ids) + self.pos_embedding(pos_ids)
        # Transformer expects (seq_len, batch_size, d_model)
        x = x.permute(1, 0, 2)
        mask = self._generate_square_subsequent_mask(seq_len).to(device)
        x = self.transformer(x, mask=mask)
        x = x.permute(1, 0, 2)
        logits = self.fc(x)
        return logits