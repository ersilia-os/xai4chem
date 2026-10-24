import numpy as np
from rdkit import Chem, DataStructs

from xai4chem.featurizers.morgan import morgan_bitvect
from xai4chem.similarity import fingerprints, nearest

from .conftest import make_library


def test_nearest_matches_rdkit():
    smiles = make_library()[:120]
    queries, reference = smiles[:40], smiles[40:]
    T, idx = nearest(fingerprints(queries), fingerprints(reference))
    ref_fps = [morgan_bitvect(Chem.MolFromSmiles(s)) for s in reference]
    for i, q in enumerate(queries):
        sims = np.array(DataStructs.BulkTanimotoSimilarity(morgan_bitvect(Chem.MolFromSmiles(q)), ref_fps))
        assert abs(sims.max() - T[i]) < 1e-6
        assert abs(sims[idx[i]] - sims.max()) < 1e-6
