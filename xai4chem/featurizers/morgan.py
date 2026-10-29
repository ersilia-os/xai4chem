"""Morgan count fingerprints, identical to Olinda's qmfp (radius 3, 2048 bits, counts clipped at 15).

Identical means bit for bit, so models and libraries are comparable across the two tools. Olinda measured that clipping counts at 15 loses nothing a model learns from (counts
above 15 are 0.03% of nonzero slots).
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

RADIUS = 3
NBITS = 2048
COUNT_CLIP = 15

_GENERATOR = None


def _generator():
    global _GENERATOR
    if _GENERATOR is None:
        _GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=NBITS)
    return _GENERATOR


def morgan_featurizer(smiles):
    gen = _generator()
    X = np.zeros((len(smiles), NBITS), dtype=np.uint8)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            X[i, :] = np.minimum(gen.GetCountFingerprintAsNumPy(mol), COUNT_CLIP)
    return X


def morgan_bit_info(mol):
    """Bit -> ((center_atom, radius), ...) for every bit set in ``mol``."""
    ao = rdFingerprintGenerator.AdditionalOutput()
    ao.AllocateBitInfoMap()
    _generator().GetCountFingerprint(mol, additionalOutput=ao)
    return {bit: tuple(occ) for bit, occ in ao.GetBitInfoMap().items()}


def _atom_label(atom):
    """A radius-0 environment as SMARTS: element, hydrogens, charge, degree and ring membership.

    These are the Morgan atom invariants, so different radius-0 bits get different names (a bare "C"
    would name half a dozen bits).
    """
    sym = atom.GetSymbol().lower() if atom.GetIsAromatic() else atom.GetSymbol()
    h = atom.GetTotalNumHs()
    hs = f"H{h}" if h > 1 else ("H" if h == 1 else "")
    q = atom.GetFormalCharge()
    charge = "" if q == 0 else ("+" if q == 1 else "-" if q == -1 else f"{q:+d}")
    ring = ";R" if atom.IsInRing() else ""
    return f"[{sym}{hs}{charge};D{atom.GetDegree()}{ring}]"


def fragment_smiles(mol, center, radius):
    """The substructure that sets a Morgan bit: an atom environment as SMILES (SMARTS for one atom)."""
    if radius == 0:
        return _atom_label(mol.GetAtomWithIdx(center))
    bonds = list(Chem.FindAtomEnvironmentOfRadiusN(mol, radius, center))
    if not bonds:
        return _atom_label(mol.GetAtomWithIdx(center))
    atoms = set()
    for b in bonds:
        bond = mol.GetBondWithIdx(b)
        atoms.update((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
    return Chem.MolFragmentToSmiles(mol, atomsToUse=sorted(atoms), bondsToUse=bonds)


def morgan_bitvect(mol):
    """Binary fingerprint with the same parameters, for Tanimoto similarity."""
    return _generator().GetFingerprint(mol)


class MorganFingerprint(object):
    """Morgan count fingerprint, radius 3, 2048 bits, counts clipped at 15 (Olinda's qmfp)."""

    name = "morgan"
    sparse = True
    description = "Morgan counts · radius 3 · 2048 bits · clipped at 15"

    def __init__(self):
        self.feature_names = ["bit-{0}".format(i) for i in range(NBITS)]
        self.bit_dictionary = None

    def fit(self, smiles=None):
        """Learn any data-dependent state (see :mod:`xai4chem.featurizers`); returns ``self``."""
        return self

    def transform(self, smiles):
        """Featurize ``smiles`` into a float32 matrix, one row per molecule."""
        return morgan_featurizer(smiles).astype(np.float32)

    def transform_raw(self, smiles):
        """Featurize ``smiles`` in the storage dtype (uint8 counts, float32 descriptors)."""
        return morgan_featurizer(smiles)

    def labels(self):
        """Human-readable feature names, aligned with ``feature_names``."""
        if self.bit_dictionary is None:
            return ["bit {0}".format(i) for i in range(NBITS)]
        return [self.bit_dictionary.label(i) for i in range(NBITS)]

    def atom_groups(self, mol):
        """For each feature present in ``mol``, the atom-index tuples of its occurrences."""
        # Each occurrence is attributed to its centre atom, as in xai4chem 1.0
        return {bit: [(center,) for center, _ in occ] for bit, occ in morgan_bit_info(mol).items()}

    def molecule_labels(self, mol):
        """Bit → label naming the environment that sets the bit *in this molecule*.

        Folded bits collide, so the library-wide name of a bit can be a substructure the molecule does not
        have; per-molecule outputs use this instead for the bits that are present.
        """
        out = {}
        for bit, occ in morgan_bit_info(mol).items():
            frags = list(dict.fromkeys(fragment_smiles(mol, c, r) for c, r in occ))
            out[bit] = f"bit {bit}: {frags[0]}" + (f" (+{len(frags) - 1} more)" if len(frags) > 1 else "")
        return out

    def to_config(self):
        """A JSON-serialisable description, stored in model and library manifests."""
        return {"name": self.name, "radius": RADIUS, "nbits": NBITS, "count_clip": COUNT_CLIP}

    @classmethod
    def from_config(cls, cfg):
        """Rebuild the featurizer from the dict written by ``to_config``."""
        if cfg.get("count_clip", COUNT_CLIP) != COUNT_CLIP:
            raise ValueError(
                f"This model was trained on Morgan counts clipped at {cfg.get('count_clip')}; "
                f"this xai4chem clips at {COUNT_CLIP}. Retrain the model."
            )
        return cls()
