"""Feature matrices for lists of molecules: from the reference library when possible, featurized otherwise."""

import numpy as np
import scipy.sparse as sp

from xai4chem.progress import SILENT

CHUNK = 5_000


def align_to_library(smiles, library):
    """Each molecule's library row, or -1. Fast path when the file is in library order (Olinda's layout)."""
    lib = library.smiles
    if len(smiles) == len(lib) and smiles[0] == lib[0] and smiles[-1] == lib[-1] and smiles == lib:
        return np.arange(len(smiles))
    index = library.index
    rows = np.array([index.get(s, -1) for s in smiles])
    missing = np.where(rows < 0)[0]
    if len(missing):
        # Different writers spell the same molecule differently; retry the misses as RDKit canonical SMILES
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.*")
        for i in missing:
            mol = Chem.MolFromSmiles(smiles[i])
            if mol is not None:
                rows[i] = index.get(Chem.MolToSmiles(mol), -1)
    return rows


def featurize(featurizer, smiles, display=SILENT):
    """Featurize ``smiles`` in chunks; sparse featurizers stay sparse (a dense Morgan copy is 2 GB per million)."""
    smiles = list(smiles)
    if len(smiles) <= CHUNK:
        X = featurizer.transform_raw(smiles)
        return sp.csr_matrix(X, dtype=np.float32) if featurizer.sparse else X.astype(np.float32)
    blocks = []
    with display.loop(f"featurizing ({featurizer.name})", len(smiles)) as tick:
        for start in range(0, len(smiles), CHUNK):
            block = featurizer.transform_raw(smiles[start : start + CHUNK])
            blocks.append(sp.csr_matrix(block, dtype=np.float32) if featurizer.sparse else block.astype(np.float32))
            tick(start + len(block))
    return sp.vstack(blocks, format="csr") if featurizer.sparse else np.vstack(blocks)


def load_features(smiles, rows, library, featurizer, display=SILENT):
    """Feature rows for ``smiles``: library rows where ``rows >= 0``, featurized otherwise; input order kept."""
    found = rows >= 0
    if featurizer.sparse:
        n = int(found.sum())
        if n > 200_000:
            with display.loop(f"reading {featurizer.name} features", n) as tick:
                X_lib = library.csr(featurizer.name, rows[found], tick=tick)
        else:
            X_lib = library.csr(featurizer.name, rows[found])
    else:
        X_lib = library.matrix(featurizer.name, rows[found]).astype(np.float32)
    if found.all():
        return X_lib
    missing = [s for s, f in zip(smiles, found) if not f]
    display.note(f"featurizing {len(missing):,} molecules that are not in the library")
    X_new = featurize(featurizer, missing, display)
    order = np.concatenate([np.where(found)[0], np.where(~found)[0]])
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    if featurizer.sparse:
        return sp.vstack([X_lib, X_new], format="csr")[inverse]
    return np.vstack([X_lib, X_new])[inverse]
