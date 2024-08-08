import numpy as np
import pandas as pd
import joblib
from rdkit.Chem import rdFingerprintGenerator
from sklearn.feature_selection import VarianceThreshold

from rdkit.Chem import rdMolDescriptors as rd
from rdkit import Chem

RADIUS = 3
NBITS = 2048
DTYPE = np.uint8

def clip_sparse(vect, nbits):
    l = [0]*nbits
    for i,v in vect.GetNonzeroElements().items():
        l[i] = v if v < 255 else 255
    return l


class _Fingerprinter(object):

    def __init__(self):
        self.nbits = NBITS
        self.radius = RADIUS

    def calc(self, mol):
        v = rd.GetHashedMorganFingerprint(mol, radius=self.radius, nBits=self.nbits)
        return clip_sparse(v, self.nbits)


def morgan_featurizer(smiles):
    d = _Fingerprinter()
    X = np.zeros((len(smiles), NBITS), dtype=np.int8)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        X[i,:] = d.calc(mol)
    return X


class MorganFingerprint(object):

    def __init__(self):
        pass

    def fit(self, smiles):
        X = morgan_featurizer(smiles)
        self.features = ["fp-{0}".format(i) for i in range(X.shape[1])]
        return pd.DataFrame(X, columns=self.features)

    def transform(self, smiles):
        X = morgan_featurizer(smiles)
        return pd.DataFrame(X, columns=self.features)
    
    def save(self, file_name):
        joblib.dump(self, file_name)
        
    @classmethod
    def load(cls, file_name):
        return joblib.load(file_name)