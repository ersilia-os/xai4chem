"""A synthetic reference library, teacher and labelled set, small enough for the suite to run in seconds."""

import itertools

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

SCAFFOLDS = [
    "c1ccc({})cc1",
    "c1ccc({})nc1",
    "C1CCN(CC1){}",
    "c1ccc2[nH]c({})cc2c1",
    "O=C(N{})c1ccccc1",
    "c1csc({})n1",
]
R_GROUPS = [
    "C", "CC", "CCC", "CO", "CN", "C(=O)O", "C(=O)N", "Cl", "F", "Br", "O", "N", "OC", "C#N",
    "[N+](=O)[O-]", "S(=O)(=O)N", "C(F)(F)F", "CCO", "CCN", "c1ccccc1", "C1CC1", "NC(=O)C", "OCC", "CCCl",
]


def make_library():
    smiles = []
    for scaffold, (r1, r2) in itertools.product(SCAFFOLDS, itertools.combinations(R_GROUPS, 2)):
        smi = scaffold.format(r1 + r2) if "N(" not in scaffold else scaffold.format(r1)
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            smiles.append(Chem.MolToSmiles(mol))
    return list(dict.fromkeys(smiles))


AMIDE = Chem.MolFromSmarts("C(=O)N")
HALOGEN = Chem.MolFromSmarts("[Cl,Br,F]")
NITRO = Chem.MolFromSmarts("[N+](=O)[O-]")


def teacher(smiles):
    """A probability that rises with amides and halogens."""
    out = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        z = 1.5 * m.HasSubstructMatch(AMIDE) + 1.0 * len(m.GetSubstructMatches(HALOGEN)) - 1.0
        out.append(1 / (1 + np.exp(-z)))
    return np.array(out)


def experiments(smiles):
    """Hard labels that mostly agree with the teacher, except that nitro groups also count."""
    m = [Chem.MolFromSmiles(s) for s in smiles]
    z = np.log(teacher(smiles) / (1 - teacher(smiles))) + np.array([2.0 * x.HasSubstructMatch(NITRO) for x in m])
    return (z > 0).astype(int)


@pytest.fixture(scope="session")
def workspace(tmp_path_factory):
    root = tmp_path_factory.mktemp("xai4chem")
    smiles = make_library()
    pd.DataFrame({"smiles": smiles}).to_csv(root / "library_input.csv", index=False)
    rng = np.random.default_rng(0)
    soft = teacher(smiles)
    soft2 = np.clip(soft + rng.normal(0, 0.05, len(soft)), 0, 1)
    pd.DataFrame({"smiles": smiles, "activity": soft, "noisy": soft2}).to_csv(root / "soft.csv", index=False)
    take = rng.choice(len(smiles), size=150, replace=False)
    hs = [smiles[i] for i in take]
    pd.DataFrame({"smiles": hs, "activity": experiments(hs)}).to_csv(root / "hard.csv", index=False)
    return root, smiles
