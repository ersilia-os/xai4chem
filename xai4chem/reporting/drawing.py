"""Molecule drawings: attributions mapped onto atoms, rendered with RDKit (SVG for pages, PNG for files)."""

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Draw, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

RDLogger.DisableLog("rdApp.*")

# Ersilia article palette: crimson raises the prediction, cobalt lowers it
POS = (0.902, 0.224, 0.275)
NEG = (0.271, 0.482, 0.616)
SVG_SIZE = 260
PNG_SIZE = 400


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


def _color(s, vmax):
    t = min(1.0, abs(s) / vmax) if vmax > 0 else 0.0
    base = POS if s > 0 else NEG
    # blend from white towards the hue; keep a floor so small effects stay visible
    a = 0.15 + 0.85 * t
    return tuple(1 - a * (1 - c) for c in base)


def _highlights(mol, scores, vmax, threshold=0.04):
    atom_cols = {i: _color(s, vmax) for i, s in enumerate(scores) if vmax > 0 and abs(s) / vmax > threshold}
    bond_cols = {}
    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a1 in atom_cols and a2 in atom_cols:
            bond_cols[bond.GetIdx()] = atom_cols[a1] if abs(scores[a1]) >= abs(scores[a2]) else atom_cols[a2]
    return atom_cols, bond_cols


def prepare(mol):
    mol = Chem.Mol(mol)
    rdDepictor.SetPreferCoordGen(True)
    AllChem.Compute2DCoords(mol)
    return mol


def atom_svg(mol, scores, vmax, size=SVG_SIZE):
    """One molecule, atoms tinted by score on a shared scale, as an inline SVG string."""
    atom_cols, bond_cols = _highlights(mol, scores, vmax)
    drawer = rdMolDraw2D.MolDraw2DSVG(size, size)
    opts = drawer.drawOptions()
    opts.useBWAtomPalette()
    opts.clearBackground = False
    opts.highlightRadius = 0.35
    drawer.DrawMolecule(
        mol,
        highlightAtoms=list(atom_cols),
        highlightAtomColors=atom_cols,
        highlightBonds=list(bond_cols),
        highlightBondColors=bond_cols,
    )
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return svg[svg.index("<svg") :]


def plain_svg(mol, size=SVG_SIZE):
    drawer = rdMolDraw2D.MolDraw2DSVG(size, size)
    drawer.drawOptions().clearBackground = False
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return svg[svg.index("<svg") :]


def atom_png(mol, panels, output_path, vmax):
    """Several views of one molecule side by side (``panels`` = [(legend, scores)]), as a PNG file."""
    n = len(panels)
    drawer = rdMolDraw2D.MolDraw2DCairo(PNG_SIZE * n, PNG_SIZE, PNG_SIZE, PNG_SIZE)
    drawer.drawOptions().useBWAtomPalette()
    hl_atoms, hl_atom_cols, hl_bonds, hl_bond_cols = [], [], [], []
    for _, scores in panels:
        ac, bc = _highlights(mol, scores, vmax)
        hl_atoms.append(list(ac))
        hl_atom_cols.append(ac)
        hl_bonds.append(list(bc))
        hl_bond_cols.append(bc)
    drawer.DrawMolecules(
        [mol] * n,
        highlightAtoms=hl_atoms,
        highlightAtomColors=hl_atom_cols,
        highlightBonds=hl_bonds,
        highlightBondColors=hl_bond_cols,
        legends=[legend for legend, _ in panels],
    )
    drawer.FinishDrawing()
    drawer.WriteDrawingText(str(output_path))


def fragment_svg(smiles_or_smarts, size=150, smarts=False):
    """A small drawing of a substructure; empty for single atoms, whose label already says it all."""
    if not smarts and ";" in smiles_or_smarts:
        return ""  # a single-atom label such as [cH;D2;R], not a SMILES
    mol = Chem.MolFromSmarts(smiles_or_smarts) if smarts else Chem.MolFromSmiles(smiles_or_smarts, sanitize=False)
    if mol is None or mol.GetNumAtoms() < 2:
        return ""
    try:
        mol.UpdatePropertyCache(strict=False)
        AllChem.Compute2DCoords(mol)
    except Exception:
        return ""
    drawer = rdMolDraw2D.MolDraw2DSVG(size, size)
    opts = drawer.drawOptions()
    opts.clearBackground = False
    opts.maxFontSize = 14
    opts.padding = 0.12
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return svg[svg.index("<svg") :]


__all__ = ["atom_attributions", "atom_svg", "plain_svg", "atom_png", "fragment_svg", "prepare", "Draw"]
