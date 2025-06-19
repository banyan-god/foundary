import sentencepiece as spm

class SPTokenizer:
    """Wrapper around SentencePieceProcessor"""
    def __init__(self, model_file: str):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_file)
        # remember original model and vocab paths for saving
        self.model_file = model_file
        # assume vocab file has same prefix with .vocab extension
        if model_file.endswith('.model'):
            self.vocab_file = model_file[:-6] + '.vocab'
        else:
            self.vocab_file = model_file + '.vocab'

    def encode(self, text: str):
        return self.sp.encode(text, out_type=int)

    def save(self, path_prefix: str):
        """
        Save the SentencePiece model and vocab files to the given path prefix.
        Writes {path_prefix}.model and {path_prefix}.vocab
        """
        import shutil
        # copy model file
        try:
            shutil.copy(self.model_file, f"{path_prefix}.model")
        except Exception:
            pass
        # copy vocab file
        try:
            shutil.copy(self.vocab_file, f"{path_prefix}.vocab")
        except Exception:
            pass
    def decode(self, ids: list[int]) -> str:
        return self.sp.decode(ids)

    @classmethod
    def train(cls, input_file: str, model_prefix: str, vocab_size: int = 1000):
        # ensure vocab size does not exceed number of training samples to avoid SP errors
        effective_vocab = vocab_size
        try:
            with open(input_file, encoding='utf-8') as f:
                lines = [l for l in f.read().splitlines() if l.strip()]
            if lines:
                effective_vocab = min(vocab_size, len(lines))
        except Exception:
            pass
        spm.SentencePieceTrainer.Train(
            input=input_file,
            model_prefix=model_prefix,
            vocab_size=effective_vocab,
            character_coverage=1.0,
            model_type='unigram'
        )
        return cls(f"{model_prefix}.model")
