"""1-NN Tanimoto similarity on binary Morgan fingerprints, as matrix products.

T, the gate's input, is the similarity of a query to its nearest labelled molecule. Computing it as
``|a ∧ b| / (|a| + |b| − |a ∧ b|)`` from one matrix product per block of queries is orders of magnitude
faster than per-molecule RDKit loops, which matters when the gate is evaluated across the whole library.

The intersections of drug-like fingerprints are almost all nonzero (99.9% measured on the Ersilia library),
so the product is computed as a *dense* float32 BLAS matrix product, about 8x faster than a sparse one.
For very large labelled sets, whose dense copy would not fit comfortably in memory, the sparse product is
used instead. The fingerprint is the binary version of the Morgan counts (radius 3, 2048 bits), whatever
featurizer the model uses, so T means the same thing in every model.
"""

import numpy as np
import scipy.sparse as sp

from xai4chem.featurizers.morgan import morgan_featurizer

BLOCK_CELLS = 20_000_000
LIBRARY_BLOCK = 200_000
DENSE_REFERENCE_BYTES = 600_000_000


def binarize(X):
    """Counts (dense or CSR) → binary float32 CSR."""
    X = X if sp.issparse(X) else sp.csr_matrix(np.asarray(X))
    X = X.tocsr().astype(np.float32)
    X.eliminate_zeros()
    X.data[:] = 1.0
    return X


def fingerprints(smiles):
    """Binary Morgan fingerprints of SMILES, as the gate's Tanimoto sees them."""
    return binarize(morgan_featurizer(list(smiles)))


def nearest(queries, reference):
    """For each query row, the max Tanimoto to any reference row, and that row's index."""
    Q, R = binarize(queries), binarize(reference)
    n_ref = R.shape[0]
    T = np.zeros(Q.shape[0], dtype=np.float32)
    idx = np.full(Q.shape[0], -1, dtype=np.int64)
    if n_ref == 0 or Q.shape[0] == 0:
        return T, idx
    q_pop = np.asarray(Q.sum(1), dtype=np.float32).ravel()
    r_pop = np.asarray(R.sum(1), dtype=np.float32).ravel()
    dense = n_ref * R.shape[1] * 4 <= DENSE_REFERENCE_BYTES
    RT = np.ascontiguousarray(R.toarray().T) if dense else R.T.tocsc()
    block = max(1, BLOCK_CELLS // n_ref)
    for start in range(0, Q.shape[0], block):
        stop = min(start + block, Q.shape[0])
        if dense:
            inter = Q[start:stop].toarray() @ RT
        else:
            inter = (Q[start:stop] @ RT).toarray()
        union = q_pop[start:stop, None] + r_pop[None, :]
        union -= inter
        np.divide(inter, union, out=inter, where=union > 0)
        inter[union <= 0] = 0.0
        j = inter.argmax(1)
        idx[start:stop] = j
        T[start:stop] = inter[np.arange(stop - start), j]
    return T, idx


def nearest_large(queries, reference, block=64):
    """:func:`nearest` against a very large reference (the whole library), as reference × queries.

    A sparse library times a small dense block of queries is cheap (about 1 s per 100 queries against 1.36M
    molecules); transposing or densifying the library is not. Blocks of the library are reduced one at a time,
    so the full library × queries matrix never exists.
    """
    Q = binarize(queries)
    R = reference if _is_binary_csr(reference) else binarize(reference)
    T = np.zeros(Q.shape[0], dtype=np.float32)
    idx = np.full(Q.shape[0], -1, dtype=np.int64)
    bounds = list(range(0, R.shape[0], LIBRARY_BLOCK)) + [R.shape[0]]
    parts = [(R[a:b], np.asarray(R[a:b].sum(1), dtype=np.float32).ravel(), a) for a, b in zip(bounds[:-1], bounds[1:])]
    for start in range(0, Q.shape[0], block):
        Qd = Q[start : start + block].toarray().T
        q_pop = Qd.sum(0)
        best = np.full(Qd.shape[1], -1.0, dtype=np.float32)
        where = np.full(Qd.shape[1], -1, dtype=np.int64)
        for Rb, r_pop, offset in parts:
            inter = np.asarray(Rb @ Qd, dtype=np.float32)
            union = r_pop[:, None] + q_pop[None, :]
            union -= inter
            np.divide(inter, union, out=inter, where=union > 0)
            inter[union <= 0] = 0.0
            j = inter.argmax(0)
            sims = inter[j, np.arange(inter.shape[1])]
            better = sims > best
            best[better], where[better] = sims[better], j[better] + offset
        T[start : start + block] = best
        idx[start : start + block] = where
    return T, idx


def _is_binary_csr(X):
    return sp.issparse(X) and X.format == "csr" and X.dtype == np.float32 and (X.nnz == 0 or X.data.max() <= 1)
