"""1-NN Tanimoto similarity on binary Morgan fingerprints, as sparse matrix products.

T, the gate's input, is the similarity of a query to its nearest labelled molecule. Computing it as
``|a ∧ b| / (|a| + |b| − |a ∧ b|)`` with one sparse product per block of queries is orders of magnitude
faster than per-molecule RDKit loops, which matters when the gate is evaluated across the library panel.
The fingerprint is the binary version of the Morgan counts (radius 3, 2048 bits), whatever featurizer
the model uses, so T means the same thing in every model.
"""

import numpy as np
import scipy.sparse as sp

from xai4chem.featurizers.morgan import morgan_featurizer

BLOCK_CELLS = 20_000_000


def binarize(X):
    """Counts (dense or CSR) → binary float32 CSR."""
    X = X if sp.issparse(X) else sp.csr_matrix(np.asarray(X))
    X = X.tocsr().astype(np.float32)
    X.eliminate_zeros()
    X.data[:] = 1.0
    return X


def fingerprints(smiles):
    return binarize(morgan_featurizer(list(smiles)))


def nearest(queries, reference):
    """For each query row, the max Tanimoto to any reference row, and that row's index."""
    Q, R = binarize(queries), binarize(reference)
    n_ref = R.shape[0]
    T = np.zeros(Q.shape[0], dtype=np.float32)
    idx = np.full(Q.shape[0], -1, dtype=np.int64)
    if n_ref == 0 or Q.shape[0] == 0:
        return T, idx
    q_pop = np.asarray(Q.sum(1)).ravel()
    r_pop = np.asarray(R.sum(1)).ravel()
    RT = R.T.tocsc()
    block = max(1, BLOCK_CELLS // n_ref)
    for start in range(0, Q.shape[0], block):
        stop = min(start + block, Q.shape[0])
        inter = (Q[start:stop] @ RT).toarray()
        union = q_pop[start:stop, None] + r_pop[None, :] - inter
        with np.errstate(invalid="ignore", divide="ignore"):
            sim = np.where(union > 0, inter / union, 0.0)
        j = sim.argmax(1)
        idx[start:stop] = j
        T[start:stop] = sim[np.arange(stop - start), j]
    return T, idx
