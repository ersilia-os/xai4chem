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
DENSE_REFERENCE_BYTES = 600_000_000


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


def nearest_large(queries, reference, block=64, threads=None):
    """:func:`nearest` against a very large reference (the whole library), as reference × queries.

    The library is split into row blocks that threads multiply by a small dense block of queries (scipy
    releases the GIL there, ~3x faster on 10 cores), and each thread reduces its block to a per-query
    maximum, so the full library × queries matrix never exists.
    """
    import os
    from concurrent.futures import ThreadPoolExecutor

    Q = binarize(queries)
    already_binary = sp.issparse(reference) and reference.format == "csr" and reference.dtype == np.float32
    R = reference if already_binary and (reference.nnz == 0 or reference.data.max() <= 1) else binarize(reference)
    threads = threads or min(8, os.cpu_count() or 1)
    bounds = np.linspace(0, R.shape[0], 2 * threads + 1).astype(int)
    parts = [
        (R[a:b], np.asarray(R[a:b].sum(1), dtype=np.float32).ravel(), a)
        for a, b in zip(bounds[:-1], bounds[1:])
        if b > a
    ]
    T = np.zeros(Q.shape[0], dtype=np.float32)
    idx = np.full(Q.shape[0], -1, dtype=np.int64)

    def best(part, Qd, q_pop):
        Rb, r_pop, offset = part
        inter = np.asarray(Rb @ Qd, dtype=np.float32)
        union = r_pop[:, None] + q_pop[None, :]
        union -= inter
        np.divide(inter, union, out=inter, where=union > 0)
        inter[union <= 0] = 0.0
        j = inter.argmax(0)
        return inter[j, np.arange(inter.shape[1])], j + offset

    with ThreadPoolExecutor(threads) as pool:
        for start in range(0, Q.shape[0], block):
            Qd = Q[start : start + block].toarray().T
            q_pop = Qd.sum(0)
            results = list(pool.map(lambda part: best(part, Qd, q_pop), parts))
            sims = np.stack([r[0] for r in results])
            rows = np.stack([r[1] for r in results])
            k = sims.argmax(0)
            cols = np.arange(sims.shape[1])
            T[start : start + block] = sims[k, cols]
            idx[start : start + block] = rows[k, cols]
    return T, idx
