import os
import shutil
import pytest

from models.tokenizer import SPTokenizer
import services.model_manager as mgr
from config import Config



@pytest.fixture()
def clean_state(tmp_path, monkeypatch):
    # Redirect SP model prefix and train data to temp
    prefix = tmp_path / 'spm'
    monkeypatch.setattr(Config, 'SP_MODEL_PREFIX', str(prefix))
    # ensure no files exist
    for ext in ['.model', '.vocab']:
        p = str(prefix) + ext
        if os.path.exists(p):
            os.remove(p)
    # no train data
    monkeypatch.setattr(Config, 'SP_TRAIN_DATA', None)
    # inject dummy tokenizer
    monkeypatch.setattr(mgr, 'SPTokenizer', DummySPTokenizer)
    # avoid copying default files by ensuring they don't exist
    if os.path.exists('spm.model'): os.remove('spm.model')
    if os.path.exists('spm.vocab'): os.remove('spm.vocab')
    yield




def test_load_all_with_train_data(monkeypatch):
    # Create fake train data file so train branch is used
    td = 'train.txt'
    with open(td, 'w') as f:
        f.write('dummy')
    monkeypatch.setattr(Config, 'SP_TRAIN_DATA', td)
    # Now load_all should call DummySPTokenizer.train and succeed
    mgr.tokenizer = None
    mgr.model = None
    mgr.load_all()
    # after loading, tokenizer should be set
    assert isinstance(mgr.tokenizer, SPTokenizer)
    # model should be instantiated
    assert mgr.model is not None

def test_classification_mapping():
    # Setup example_map
    mgr._example_map.clear()
    # define a mapping key based on sorted dict items
    example = {'a': 'x', 'b': 'y'}
    key = tuple(sorted(example.items()))
    mgr._example_map[key] = 'label1'
    # create dummy request
    from schemas.transaction import InferenceRequest, Transaction
    tx = Transaction(description='x', name='n', merchant='m', amount='1.0')
    req = InferenceRequest(current_transaction=tx, user_history=[])
    # monkeypatch model to ensure _example_map path
    # duplicate mapping for actual dict
    # override model to skip load
    monkeypatch = pytest.MonkeyPatch()

    res = mgr.predict(req)
    # if no exact match, default should come from _example_map values
    assert res.predicted_category in mgr._example_map.values()
    assert res.confidence in (0.0, 1.0)
    monkeypatch.undo()