"""XAI4Chem: explainable surrogates of chemistry models, learned from soft labels.

Heavy dependencies (xgboost, rdkit) are imported lazily, so that ``xai4chem --help`` and
``import xai4chem`` stay fast.
"""

__version__ = "2.0.0"

_LAZY = {
    "setup": "xai4chem.api",
    "fit": "xai4chem.api",
    "explain": "xai4chem.api",
    "predict": "xai4chem.api",
    "XAIModel": "xai4chem.models.model",
    "ReferenceLibrary": "xai4chem.data.library",
    "get_featurizer": "xai4chem.featurizers",
    "MorganFingerprint": "xai4chem.featurizers",
    "AccFgFingerprint": "xai4chem.featurizers",
    "DatamolDescriptor": "xai4chem.featurizers",
}


def __getattr__(name):
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module 'xai4chem' has no attribute {name!r}")
