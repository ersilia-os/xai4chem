import datamol as dm
import numpy as np
import pandas as pd

MAX_NA = 0.1


def datamol_featurizer(smiles_list):
    R = []
    for smiles in smiles_list:
        mol = dm.to_mol(smiles)
        R.append(dm.descriptors.compute_many_descriptors(mol) if mol is not None else {})
    return pd.DataFrame(R)


class DatamolDescriptor(object):
    """Datamol's global physicochemical descriptors.

    Fitting drops descriptors with too many missing values or no variance, and learns median imputation.
    xai4chem 1.0 also discretized descriptors into quantile bins; raw values are kept now, because SHAP dependence on a
    raw descriptor (e.g. logP) is what a chemist can read.
    """

    name = "datamol"
    sparse = False
    description = "datamol physicochemical descriptors"

    def __init__(self, columns=None, medians=None):
        self.columns = columns
        self.medians = medians
        self.feature_names = list(columns) if columns is not None else None

    def fit(self, smiles):
        """Learn any data-dependent state (see :mod:`xai4chem.featurizers`); returns ``self``."""
        df = datamol_featurizer(smiles)
        X = df.to_numpy(dtype=np.float32)
        keep = []
        for j, col in enumerate(df.columns):
            v = X[:, j]
            ok = ~np.isnan(v)
            if ok.mean() < 1 - MAX_NA or np.nanstd(v) == 0:
                continue
            keep.append((col, float(np.median(v[ok]))))
        self.columns = [c for c, _ in keep]
        self.medians = [m for _, m in keep]
        self.feature_names = list(self.columns)
        return self

    def transform(self, smiles):
        """Featurize ``smiles`` into a float32 matrix, one row per molecule."""
        if self.columns is None:
            raise ValueError("The datamol featurizer has not been fitted.")
        df = datamol_featurizer(smiles).reindex(columns=self.columns)
        X = df.to_numpy(dtype=np.float32, copy=True)  # pandas 3 hands out read-only views
        for j, m in enumerate(self.medians):
            X[np.isnan(X[:, j]), j] = m
        return X

    def transform_raw(self, smiles):
        """Featurize ``smiles`` in the storage dtype (uint8 counts, float32 descriptors)."""
        return self.transform(smiles)

    def labels(self):
        """Human-readable feature names, aligned with ``feature_names``."""
        return list(self.columns)

    def atom_groups(self, mol):
        """For each feature present in ``mol``, the atom-index tuples of its occurrences."""
        return None

    def molecule_labels(self, mol):
        """Feature labels do not depend on the molecule for this featurizer."""
        return {}

    def to_config(self):
        """A JSON-serialisable description, stored in model and library manifests."""
        return {"name": self.name, "columns": self.columns, "medians": self.medians}

    @classmethod
    def from_config(cls, cfg):
        """Rebuild the featurizer from the dict written by ``to_config``."""
        return cls(columns=cfg["columns"], medians=cfg["medians"])
