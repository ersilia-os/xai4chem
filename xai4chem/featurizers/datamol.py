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
    v1 also discretized descriptors into quantile bins; v2 keeps raw values, because SHAP dependence on a
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
        if self.columns is None:
            raise ValueError("The datamol featurizer has not been fitted.")
        df = datamol_featurizer(smiles).reindex(columns=self.columns)
        X = df.to_numpy(dtype=np.float32)
        for j, m in enumerate(self.medians):
            X[np.isnan(X[:, j]), j] = m
        return X

    def transform_raw(self, smiles):
        return self.transform(smiles)

    def labels(self):
        return list(self.columns)

    def atom_groups(self, mol):
        return None

    def to_config(self):
        return {"name": self.name, "columns": self.columns, "medians": self.medians}

    @classmethod
    def from_config(cls, cfg):
        return cls(columns=cfg["columns"], medians=cfg["medians"])
