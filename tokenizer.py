
import torch.nn as nn
import json
import re


class SimpleTokenizer:
    def __init__(self, vocab=None):
        self.vocab = vocab or {}
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.unk_token = '[UNK]'
        self.cls_token = '[CLS]'
        self.sep_token = '[SEP]'
        self.pad_token = '[PAD]'
        self.special_tokens = [self.unk_token, self.cls_token, self.sep_token, self.pad_token]

    def build_vocab(self, texts, min_freq=1):
        from collections import Counter
        counter = Counter()
        for text in texts:
            tokens = self.tokenize(text)
            counter.update(tokens)
        idx = 0
        for token in self.special_tokens:
            self.vocab[token] = idx
            idx += 1
        for token, freq in counter.items():
            if freq >= min_freq and token not in self.vocab:
                self.vocab[token] = idx
                idx += 1
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def grow_vocab(self, texts, min_freq=1):
        from collections import Counter
        counter = Counter()
        for text in texts:
            tokens = self.tokenize(text)
            counter.update(tokens)
        idx = max(self.vocab.values(), default=-1) + 1
        for token, freq in counter.items():
            if freq >= min_freq and token not in self.vocab:
                self.vocab[token] = idx
                idx += 1
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def tokenize(self, text):
        return re.findall(r'\w+|\S', text.lower())

    def encode(self, text, max_length=128):
        tokens = [self.cls_token] + self.tokenize(text)[:max_length-2] + [self.sep_token]
        ids = [self.vocab.get(token, self.vocab[self.unk_token]) for token in tokens]
        if len(ids) < max_length:
            ids += [self.vocab[self.pad_token]] * (max_length - len(ids))
        return ids

    def decode(self, ids):
        return ' '.join([self.inv_vocab.get(i, self.unk_token) for i in ids])

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.vocab, f)

    @classmethod
    def load(cls, path):
        with open(path, "r") as f:
            vocab = json.load(f)
        return cls(vocab)