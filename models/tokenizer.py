import sentencepiece as spm

class SPTokenizer:
    """Wrapper around SentencePieceProcessor"""
    def __init__(self, model_file: str):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_file)

    def encode(self, text: str):
        return self.sp.encode(text, out_type=int)

    def save(self, path_prefix: str):
        pass

    @classmethod
    def train(cls, input_file: str, model_prefix: str, vocab_size: int = 1000):
        spm.SentencePieceTrainer.Train(
            input=input_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            character_coverage=1.0,
            model_type='unigram'
        )
        return cls(f"{model_prefix}.model")
