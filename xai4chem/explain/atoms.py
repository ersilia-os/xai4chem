"""From feature attributions to atom attributions."""

import numpy as np


def atom_attributions(featurizer, mol, phi_row):
    """Distribute feature attributions onto atoms.

    Each feature's value is split evenly over its occurrences in the molecule, and each occurrence's share
    is split evenly over its atoms. So the atom scores sum exactly to the attribution of the features
    present. What is left (``unmapped``) is the effect of features *absent* from the molecule, which
    TreeSHAP also scores but which has no atoms to sit on. Returns ``None`` for featurizers whose features
    do not map to atoms.
    """
    groups = featurizer.atom_groups(mol)
    if groups is None:
        return None
    scores = np.zeros(mol.GetNumAtoms())
    mapped = 0.0
    for f, occurrences in groups.items():
        v = float(phi_row[f])
        if v == 0 or not occurrences:
            continue
        share = v / len(occurrences)
        for atoms in occurrences:
            for a in atoms:
                scores[a] += share / len(atoms)
        mapped += v
    return scores, float(phi_row.sum()) - mapped
