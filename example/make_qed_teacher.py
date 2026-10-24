"""A stand-in teacher for trying xai4chem end to end: RDKit's QED over the reference library.

QED is a smooth, substructure-driven score in [0, 1], so its explanations can be checked by eye. The hard
labels deliberately disagree with it on one point: molecules with a carboxylic acid are labelled inactive.
A model fitted with them should show that disagreement in its "experiments" and "disagreement" views.

    python make_qed_teacher.py            # writes data/erl0_qed.csv and data/hard_qed.csv
"""

from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

N_HARD = 3000


def qed_chunk(smiles):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import QED

    RDLogger.DisableLog("rdApp.*")
    out = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        try:
            out.append(QED.qed(mol) if mol is not None else np.nan)
        except Exception:
            out.append(np.nan)
    return out


def main():
    from rdkit import Chem

    from xai4chem.library import ReferenceLibrary, default_n_jobs

    out = Path(__file__).parent / "data"
    out.mkdir(exist_ok=True)
    smiles = ReferenceLibrary().smiles
    chunks = [smiles[i : i + 5000] for i in range(0, len(smiles), 5000)]
    with get_context("spawn").Pool(default_n_jobs()) as pool:
        qed = [v for part in pool.imap(qed_chunk, chunks) for v in part]
    pd.DataFrame({"smiles": smiles, "qed": qed}).to_csv(out / "erl0_qed.csv", index=False)

    acid = Chem.MolFromSmarts("C(=O)[OH]")
    rows = []
    for i in np.random.default_rng(7).choice(len(smiles), N_HARD, replace=False):
        mol = Chem.MolFromSmiles(smiles[i])
        if mol is None or not np.isfinite(qed[i]):
            continue
        rows.append((smiles[i], int(qed[i] > 0.65 and not mol.HasSubstructMatch(acid))))
    pd.DataFrame(rows, columns=["smiles", "qed"]).to_csv(out / "hard_qed.csv", index=False)
    print(f"wrote {out / 'erl0_qed.csv'} and {out / 'hard_qed.csv'}")


if __name__ == "__main__":
    main()
