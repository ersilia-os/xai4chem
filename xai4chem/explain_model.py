import shap
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D, MolDraw2DCairo
from rdkit.Chem.Draw import IPythonConsole
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


RADIUS = 3
NBITS = 2048


def explain_model(model, X, smiles_list, use_fingerprints, output_folder):
    create_output_folder(output_folder)

    explainer = shap.TreeExplainer(model)
    explanation = explainer(X)

    percentiles = [0, 25, 50, 75, 100]
    sample_indices = np.percentile(range(X.shape[0]), percentiles).astype(int)

    for i, idx in enumerate(sample_indices):
        smiles = smiles_list[idx] if smiles_list is not None else f'Sample {idx}'
        plot_waterfall(explanation, idx, smiles, output_folder, percentiles[i])

    save_shap_values_to_csv(explanation, X, X.columns, output_folder)
    plot_summary_plots(explanation, output_folder)
    plot_scatter_plots(explanation, X.columns, output_folder)

    if smiles_list is not None and use_fingerprints:
        for i, idx in enumerate(sample_indices):
            smiles = smiles_list[idx]
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                sample_shap_values = explanation[idx].values
                bit_info = {}
                AllChem.GetHashedMorganFingerprint(mol, radius=RADIUS, nBits=NBITS, bitInfo=bit_info)

                bit_shap_values = {}
                feature_names = X.columns
                for bit_idx, feature_name in enumerate(feature_names): 
                    bit = int(feature_name.split('-')[1])
                    if 0 <= bit_idx < len(sample_shap_values):
                        bit_shap_values[bit] = sample_shap_values[bit_idx]

                valid_top_bits = [bit for bit in sorted(bit_shap_values.keys(), key=lambda b: abs(bit_shap_values[b]),
                                                        reverse=True) if bit in bit_info][:5]

                draw_top_features(mol, bit_info, valid_top_bits, smiles,
                                  os.path.join(output_folder, f'sample_p{percentiles[i]}_top_features.png'))
                highlight_and_draw_molecule(mol, bit_info, valid_top_bits, bit_shap_values, smiles,
                                            os.path.join(output_folder,
                                                         f"sample_p{percentiles[i]}_shap_highlights.png"))

    return explanation


def create_output_folder(output_folder):
    """Create output folder if it does not exist."""
    os.makedirs(output_folder, exist_ok=True)


def save_shap_values_to_csv(explanation, X, feature_names, output_folder):
    """Save SHAP values and features to a CSV file."""
    shap_df = pd.DataFrame(explanation.values, columns=[f'SHAP_{name}' for name in feature_names])
    data_df = pd.DataFrame(X, columns=feature_names)
    combined_df = pd.concat([data_df, shap_df], axis=1)
    combined_df.to_csv(os.path.join(output_folder, 'shap_values.csv'), index=False)


def plot_waterfall(explanation, idx, smiles, output_folder, percentiles):
    """Create a waterfall plot for a given sample."""
    shap.waterfall_plot(explanation[idx], max_display=15, show=False)
    plt.title(f"Molecule: {smiles}")
    plt.savefig(os.path.join(output_folder, f"interpretability_sample_p{percentiles}.png"), bbox_inches='tight')
    plt.close()


def plot_summary_plots(explanation, output_folder):
    """Create summary plots: bar plot and beeswarm plot."""
    shap.plots.bar(explanation, max_display=20, show=False)
    plt.savefig(os.path.join(output_folder, "interpretability_bar_plot.png"), bbox_inches='tight')
    plt.close()

    shap.plots.beeswarm(explanation, max_display=15, show=False)
    plt.savefig(os.path.join(output_folder, "interpretability_beeswarm_plot.png"), bbox_inches='tight')
    plt.close()


def plot_scatter_plots(explanation, feature_names, output_folder):
    """Create scatter plots for the top 5 features."""
    shap_values = explanation.values
    top_features = np.argsort(-np.abs(shap_values).mean(0))[:5]

    for feature in top_features:
        shap.plots.scatter(explanation[:, feature], show=False)
        plt.savefig(os.path.join(output_folder, f"interpretability_{feature_names[feature]}.png"), bbox_inches='tight')
        plt.close()


def draw_top_features(mol, bit_info, valid_top_bits, smiles, output_path):
    """Draw and save top features(bits)."""
    list_bits = []
    legends = []

    for x in valid_top_bits:
        for i in range(len(bit_info[x])):
            list_bits.append((mol, x, bit_info, i))
            legends.append(str(x))

    drawOptions = Draw.rdMolDraw2D.MolDrawOptions()
    drawOptions.prepareMolsBeforeDrawing = False
    p = Draw.DrawMorganBits(list_bits, molsPerRow=6, legends=legends, drawOptions=drawOptions)
    p.save(output_path)
    
    add_title_to_image(output_path, f"Top 5 features for: {smiles}")


def highlight_and_draw_molecule(mol, bit_info, valid_top_bits, bit_shap_values, smiles, output_path):
    """Highlight atoms and bonds based on SHAP values and save the molecule image."""
    highlights = set()
    atom_colors = {}
    bond_highlights = set()
    bond_colors = {}

    for bit in valid_top_bits:
        shap_value = bit_shap_values.get(bit, 0)
        color = (1, 0, 0, 0.6) if shap_value > 0 else (0, 0, 1, 0.6)

        bit_atoms = bit_info.get(bit, [])
        if not isinstance(bit_atoms, list):
            bit_atoms = [bit_atoms]

        atom_indices = []
        for item in bit_atoms:
            if isinstance(item, tuple):
                if len(item) == 2:
                    atom_idx, _ = item
                    atom_indices.append(atom_idx)
                elif isinstance(item[0], tuple):
                    for sub_item in item:
                        atom_idx, _ = sub_item
                        atom_indices.append(atom_idx)
            elif isinstance(item, int):
                atom_indices.append(item)

        for atom_idx in atom_indices:
            if isinstance(atom_idx, int):
                env = Chem.FindAtomEnvironmentOfRadiusN(mol, RADIUS, atom_idx)

                atoms = set()
                bonds = set()
                for bidx in env:
                    bond = mol.GetBondWithIdx(bidx)
                    atoms.add(bond.GetBeginAtomIdx())
                    atoms.add(bond.GetEndAtomIdx())
                    bonds.add(bidx)

                highlights.update(atoms)
                bond_highlights.update(bonds)

                for atom in atoms:
                    atom_colors[atom] = color
                for bond in bonds:
                    bond_colors[bond] = color

    d = MolDraw2DCairo(500, 500)
    d.drawOptions().useBWAtomPalette()
    rdMolDraw2D.PrepareAndDrawMolecule(
        d,
        mol,
        highlightAtoms=list(highlights),
        highlightAtomColors=atom_colors,
        highlightBonds=list(bond_highlights),
        highlightBondColors=bond_colors
    )
    d.FinishDrawing()
    d.WriteDrawingText(output_path)
    
    add_title_to_image(output_path, f"{smiles}")

# Add titles
def add_title_to_image(image_path, title):
    img = mpimg.imread(image_path)
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.title(title, fontsize=16)
    plt.axis('off')  # Hide axis
    plt.savefig(image_path, bbox_inches='tight')
    plt.close()