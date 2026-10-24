"""Featurizers: SMILES to the feature matrix both S and H are trained on.

Every featurizer exposes the same small interface, so the rest of the package never branches on which
one is in use:

- ``fit(smiles)``: learn any data-dependent state (only datamol has some), return ``self``.
- ``transform(smiles)``: a float32 matrix, one row per molecule, one column per ``feature_names``.
- ``labels()``: human-readable names, aligned with ``feature_names``.
- ``atom_groups(mol)``: for each feature present in ``mol``, the atom-index tuples of its occurrences;
  ``None`` when features do not map to atoms (global descriptors).
- ``to_config()`` / ``from_config(cfg)``: a JSON-serialisable description, stored in the model manifest.
"""

from .accfg import AccFgFingerprint
from .datamol import DatamolDescriptor
from .morgan import MorganFingerprint

FEATURIZERS = {
    "morgan": MorganFingerprint,
    "accfg": AccFgFingerprint,
    "datamol": DatamolDescriptor,
}


def get_featurizer(name):
    """Return a fresh, unfitted featurizer by name."""
    if name not in FEATURIZERS:
        raise ValueError(f"Unknown featurizer {name!r}. Options: {', '.join(FEATURIZERS)}")
    return FEATURIZERS[name]()


def featurizer_from_config(cfg):
    """Rebuild a featurizer from the dict written by its ``to_config``."""
    return FEATURIZERS[cfg["name"]].from_config(cfg)
