"""Molecule drawings for the HTML pages: atoms tinted by attribution, and small substructure drawings (RDKit SVG)."""

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

RDLogger.DisableLog("rdApp.*")

# Ersilia article palette: crimson raises the prediction, cobalt lowers it
POS = (0.902, 0.224, 0.275)
NEG = (0.271, 0.482, 0.616)
SVG_SIZE = 260


# RDKit draws in pure black and greys; map them onto the Ersilia ink tokens so drawings sit in the page
_SVG_INK = {"#000000": "#2C3E50", "#191919": "#2C3E50", "#7F7F7F": "#9A93A6", "#7f7f7f": "#9A93A6"}


def _brand(svg):
    for old, new in _SVG_INK.items():
        svg = svg.replace(old, new)
    return svg


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
    return _brand(svg[svg.index("<svg") :])


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
    opts.useBWAtomPalette()
    opts.clearBackground = False
    opts.maxFontSize = 14
    opts.padding = 0.12
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return _brand(svg[svg.index("<svg") :])
