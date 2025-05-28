import torch
import torch.nn as nn

class VanillaTransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, num_classes=10, max_length=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)
        self.max_length = max_length

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        cls_output = x[0]
        logits = self.fc(cls_output)
        return logits