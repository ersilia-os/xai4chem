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


def test_nearest_large_matches_nearest():
    from xai4chem.similarity import nearest_large

    smiles = make_library()[:300]
    Q, R = fingerprints(smiles[:70]), fingerprints(smiles[70:])
    T1, i1 = nearest(Q, R)
    T2, i2 = nearest_large(Q, R, block=16)
    np.testing.assert_allclose(T1, T2, atol=1e-6)
    np.testing.assert_allclose(T1, T2[np.arange(len(T2))], atol=1e-6)
