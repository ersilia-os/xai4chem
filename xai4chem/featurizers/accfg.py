import os
from importlib import resources

import numpy as np
import pandas as pd
from rdkit import Chem

from xai4chem.tools.accfg.main import AccFG

REF_PATH = resources.files("xai4chem").joinpath("tools/accfg")


def ref_features():
    df = pd.read_csv(os.path.join(REF_PATH, "fgs_all.csv"))
    fgs_ref = {fg: i for i, fg in enumerate(df["Functional Group"].tolist())}
    return fgs_ref


def ref_smarts():
    df = pd.read_csv(os.path.join(REF_PATH, "fgs_all.csv"))
    smarts_ref = {sm: i for i, sm in enumerate(df["SMARTS Pattern"].tolist())}
    return smarts_ref


class _Matcher(object):
    """AccFG's matching (``AccFG.run_mol``) with the SMARTS compiled once.

    ``AccFG.run`` canonicalises the SMILES before matching, so its atom indices refer to a different
    molecule than ``Chem.MolFromSmiles(smiles)``. Matching on the caller's ``mol`` keeps atom indices
    aligned with the molecule we draw.
    """

    def __init__(self):
        fgs_ref = ref_features()
        self.queries = [
            (fgs_ref[name], Chem.MolFromSmarts(smarts)) for name, smarts in AccFG().dict_fgs.items() if name in fgs_ref
        ]
        self.n_features = len(fgs_ref)

    def match(self, mol):
        out = {}
        for idx, query in self.queries:
            hits = mol.GetSubstructMatches(query, uniquify=True)
            if hits:
                out[idx] = [tuple(h) for h in hits]
        return out


_MATCHER = None


def _matcher():
    global _MATCHER
    if _MATCHER is None:
        _MATCHER = _Matcher()
    return _MATCHER


def accfg_featurizer(smi_list):
    m = _matcher()
    arr = np.zeros((len(smi_list), m.n_features), dtype=np.uint8)
    for i, s in enumerate(smi_list):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        for idx, hits in m.match(mol).items():
            arr[i, idx] = min(len(hits), 255)
    return arr


class AccFgFingerprint(object):
    """Counts of AccFG functional groups and heterocycles."""

    name = "accfg"
    sparse = True
    description = "AccFG functional groups and heterocycles · counts"

    def __init__(self):
        self.feature_names = ["fg-{0}".format(i) for i in range(len(ref_features()))]

    def fit(self, smiles=None):
        """Learn any data-dependent state (see :mod:`xai4chem.featurizers`); returns ``self``."""
        return self

    def transform(self, smiles):
        """Featurize ``smiles`` into a float32 matrix, one row per molecule."""
        return accfg_featurizer(smiles).astype(np.float32)

    def transform_raw(self, smiles):
        """Featurize ``smiles`` in the storage dtype (uint8 counts, float32 descriptors)."""
        return accfg_featurizer(smiles)

    def labels(self):
        """Human-readable feature names, aligned with ``feature_names``."""
        return list(ref_features().keys())

    def smarts(self):
        """The SMARTS pattern of every functional group, aligned with ``feature_names``."""
        return list(ref_smarts().keys())

    def atom_groups(self, mol):
        """For each feature present in ``mol``, the atom-index tuples of its occurrences."""
        return _matcher().match(mol)

    def get_ref_features(self):
        """Functional-group name → feature index."""
        return ref_features()

    def get_ref_smarts(self):
        """SMARTS pattern → feature index."""
        return ref_smarts()

    def to_config(self):
        """A JSON-serialisable description, stored in model and library manifests."""
        return {"name": self.name, "n_features": len(self.feature_names)}

    @classmethod
    def from_config(cls, cfg):
        """Rebuild the featurizer from the dict written by ``to_config``."""
        return cls()
