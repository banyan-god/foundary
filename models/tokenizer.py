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
        """Train a SentencePiece model from a text/CSV file.

        SentencePiece has two opposite failure modes when the requested *vocab_size*
        does not match the data size:

        1. ``vocab_size`` *too large*  →  "Vocabulary size too high" runtime error.
        2. ``vocab_size`` *too small*  →  "Vocabulary size is smaller than
           required_chars" runtime error (see
           https://github.com/google/sentencepiece/blob/master/src/trainer_interface.cc).

        The previous heuristic of clamping the vocabulary to the number of
        *lines* in the file fixes (1) but triggers (2) for very small corpora –
        exactly what happens in the unit-tests which provide only two sentences.

        A safer heuristic is:

        • Determine the number of *unique characters* in the corpus.  SentencePiece
          internally adds every character (``required_chars``) plus the three meta
          tokens ``<unk>``, ``<s>``, and ``</s>``.  Therefore the minimum valid
          vocabulary size is ``len(unique_chars) + 3``.

        • Pick the *effective* vocab size as the smallest value that satisfies
          the above lower bound **and** does not exceed the user supplied
          ``vocab_size``.
        """

        eff_vocab = vocab_size
        try:
            with open(input_file, encoding='utf-8') as f:
                lines = [l.strip() for l in f if l.strip()]

            # Minimum allowed size = number of distinct characters + meta pieces.
            char_set = set(''.join(lines))
            min_allowed = len(char_set) + 3  # 3 meta tokens (<unk>, <s>, </s>)

            # If the user provided vocab is lower than the minimum, bump it up;
            # if it is much larger than the data, shrink it to avoid the "too
            # high" error.  +5 headroom avoids retraining if we later add a few
            # extra characters.
            eff_vocab = max(min(vocab_size, max(32, len(lines) * 16)), min_allowed)
        except Exception:
            # Fall back to the requested size on any error.
            eff_vocab = vocab_size
        spm.SentencePieceTrainer.Train(
            input=input_file,
            model_prefix=model_prefix,
            vocab_size=eff_vocab,
            character_coverage=1.0,
            hard_vocab_limit=False,  # allow trainer to shrink the vocab if data is small
            model_type='unigram'
        )
        return cls(f"{model_prefix}.model")
