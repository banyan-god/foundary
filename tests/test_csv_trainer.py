import os
import pandas as pd
import pytest

import csv_trainer


class Dummy:
    pass


@pytest.fixture(autouse=True)
def dummy_ar_train(monkeypatch):
    # Replace ar_train to simply return sequential loss values
    def fake_ar_train(texts, epochs, batch_size, lr):
        # Check inputs
        assert isinstance(texts, list)
        assert all(isinstance(t, str) for t in texts)
        return [0.5 * i for i in range(1, epochs + 1)]

    monkeypatch.setattr(csv_trainer, 'ar_train', fake_ar_train)
    yield


def test_train_from_csv(tmp_path, capsys):
    # Create a CSV file with two rows and three columns
    df = pd.DataFrame({
        'col1': ['a', 'b'],
        'col2': ['c', 'd'],
        'col3': ['e', 'f'],
    })
    csv_file = tmp_path / 'data.csv'
    df.to_csv(csv_file, index=False)
    # Call train_from_csv with 3 epochs
    losses = csv_trainer.train_from_csv(str(csv_file), epochs=3, batch_size=1, lr=0.01)
    # Expect fake losses
    assert losses == [0.5, 1.0, 1.5]
    # Capture printed output
    captured = capsys.readouterr()
    # Should print three epoch lines
    lines = [ln for ln in captured.out.splitlines() if ln.startswith('Epoch')]
    assert len(lines) == 3
    assert 'Epoch 1/3 - avg loss: 0.50' in lines[0]


def test_train_from_csv_empty(tmp_path):
    # Empty CSV should yield empty texts
    csv_file = tmp_path / 'empty.csv'
    # write header only
    csv_file.write_text('a,b,c\n')
    # Even with no rows, ar_train should be called with empty list
    # and return empty list
    def fake_ar_train(texts, epochs, batch_size, lr):
        assert texts == []
        return []
    # monkeypatch train invocation
    import csv_trainer as mod
    monkey = pytest.MonkeyPatch()
    monkey.setattr(mod, 'ar_train', fake_ar_train)
    losses = mod.train_from_csv(str(csv_file), epochs=1, batch_size=1, lr=0.1)
    assert losses == []
    monkey.undo()

def test_main_entrypoint(monkeypatch, tmp_path, capsys):
    # Create minimal CSV
    df = pd.DataFrame({'a': ['x'], 'b': ['y']})
    csv_file = tmp_path / 'd.csv'
    df.to_csv(csv_file, index=False)
    # fake train_from_csv to capture args
    calls = {}
    def fake_train(path, epochs, batch_size, lr):
        calls['path'] = path
        calls['epochs'] = epochs
        calls['batch_size'] = batch_size
        calls['lr'] = lr
        return [0.1]
    monkeypatch.setattr(csv_trainer, 'train_from_csv', fake_train)
    # simulate CLI args
    test_args = ['csv_trainer.py', str(csv_file), '--epochs', '3', '--batch-size', '2', '--lr', '0.05']
    monkeypatch.setattr('sys.argv', test_args)
    # Call main() explicitly to run training and print output
    csv_trainer.main()
    out = capsys.readouterr().out
    # Check that our fake_train was called with correct args
    assert calls['path'] == str(csv_file)
    assert calls['epochs'] == 3
    assert calls['batch_size'] == 2
    assert abs(calls['lr'] - 0.05) < 1e-6
    # We invoked main(), so train was called; output may be empty under fake_train
    assert isinstance(out, str)
  
def test_train_from_hf(monkeypatch, capsys):
    import csv_trainer as ct
    monkeypatch.setattr(ct, '_HAS_DATASETS', True)
    fake_data = [{'col1': 'x', 'col2': 'y'}, {'col1': 'a', 'col2': 'b'}]
    fake_ds = {'train': fake_data}
    monkeypatch.setattr(ct, 'load_dataset', lambda name: fake_ds)
    def fake_ar_train(texts, epochs, batch_size, lr):
        assert texts == ['x y', 'a b']
        return [0.2, 0.4]
    monkeypatch.setattr(ct, 'ar_train', fake_ar_train)
    losses = ct.train_from_hf('dummy_dataset', 'train', epochs=2, batch_size=1, lr=0.01)
    assert losses == [0.2, 0.4]
    out = capsys.readouterr().out
    assert 'Epoch 1/2 - avg loss: 0.2000' in out
    assert 'Epoch 2/2 - avg loss: 0.4000' in out

def test_train_from_hf_no_datasets(monkeypatch):
    import csv_trainer as ct
    monkeypatch.setattr(ct, '_HAS_DATASETS', False)
    with pytest.raises(ImportError):
        ct.train_from_hf('any', 'train', epochs=1, batch_size=1, lr=0.1)