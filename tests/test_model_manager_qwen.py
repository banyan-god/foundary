"""Ensure ModelManager instantiates **Qwen3Model** when configured via env."""

import importlib
import os

import pytest


@pytest.fixture()
def manager_qwen(monkeypatch, tmp_path):
    """Reload services.model_manager with MODEL_TYPE=qwen so that the manager
    builds a Qwen3 model.  We also monkey-patch the filesystem paths so no real
    model files are needed.
    """

    # Force env before module import
    monkeypatch.setenv("MODEL_TYPE", "qwen")

    # Re-import Config so that it picks up the new env var
    from importlib import reload
    import config as _cfg

    reload(_cfg)

    # Now reload the model_manager to read the updated Config
    import services.model_manager as mm

    reload(mm)

    yield mm


def test_load_all_creates_qwen_model(manager_qwen, monkeypatch):
    # Provide dummy SentencePiece model files
    sp_prefix = os.path.join(os.getcwd(), "spm_dummy_test_qwen")
    for ext in (".model", ".vocab"):
        open(f"{sp_prefix}{ext}", "w").close()

    monkeypatch.setattr(manager_qwen.Config, "SP_MODEL_PREFIX", sp_prefix)

    # Reuse DummySPTokenizer from another test to avoid SentencePiece dependency
    from tests.test_model_manager import DummySPTokenizer

    monkeypatch.setattr(manager_qwen, "SPTokenizer", DummySPTokenizer)

    # Ensure clean state
    manager_qwen.model = None
    manager_qwen.tokenizer = None

    manager_qwen.load_all()

    # Import class here to avoid heavy import when not needed
    from models.qwen3 import Qwen3Model

    assert isinstance(manager_qwen.model, Qwen3Model)