import numpy as np
import pandas as pd
import joblib
from rdkit.Chem import rdFingerprintGenerator
from sklearn.feature_selection import VarianceThreshold

from rdkit.Chem import rdMolDescriptors as rd
from rdkit import Chem

_MIN_PATH_LEN = 1
_MAX_PATH_LEN = 7
_N_BITS = 2048 

class _Fingerprinter(object):

    def __init__(self):
        self.minPathLen = _MIN_PATH_LEN
        self.maxPathLen = _MAX_PATH_LEN
        self.nbits = _N_BITS

    def _clip(self, v):
        if v > 127:
            v = 127
        return v

    def calc(self, mol):
        counts = Chem.RDKFingerprint( mol, minPath=self.minPathLen, maxPath=self.maxPathLen, fpSize=self.nbits)
        return [self._clip(c) for c in counts]


def rdkit_featurizer(smiles):
    desc = _Fingerprinter() 
    X = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        fp = np.array(desc.calc(mol), dtype=np.int8)
        X.append(fp)
    return X


class RDKitFingerprint(object):

    def __init__(self):
        pass

    def fit(self, smiles):
        X = rdkit_featurizer(smiles)
        self.features = ["fp-{0}".format(i) for i in range(_N_BITS)]
        return pd.DataFrame(X, columns=self.features)

    def transform(self, smiles):
        X = rdkit_featurizer(smiles)
        return pd.DataFrame(X, columns=self.features)
    
    def save(self, file_name):
        joblib.dump(self, file_name)
        
    @classmethod
    def load(cls, file_name):
        return joblib.load(file_name)