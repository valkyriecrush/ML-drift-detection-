"""
Regression test for the P0 import bug: `src/evaluate.py` imported
`from src.models import fit_models`, but the module is `src/modeling.py` ->
guaranteed ModuleNotFoundError. This test just imports the module (no need
to run main(), which trains models and is slow) to prove the import chain
resolves.
"""

import importlib


def test_evaluate_module_imports_without_error():
    module = importlib.import_module("src.evaluate")
    assert hasattr(module, "main")


def test_fit_models_is_reachable_from_modeling_not_models():
    import src.modeling as modeling

    assert hasattr(modeling, "fit_models")
    # src.models does not exist -- the bug was importing from this module.
    import importlib.util

    assert importlib.util.find_spec("src.models") is None
