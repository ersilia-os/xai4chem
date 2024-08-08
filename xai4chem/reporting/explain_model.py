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
_MIN_PATH_LEN = 1
_MAX_PATH_LEN = 7 


def explain_model(model, X, smiles_list, output_folder, fingerprints=None):
    print('Explaining model')
    create_output_folder(output_folder)

    explainer = shap.TreeExplainer(model)
    explanation = explainer(X)

    #Samples
    predictions = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X) 
    percentiles = [0, 25, 50, 75, 100]
    percentile_values = np.percentile(predictions, percentiles)

    # Indices of the predictions closest to the percentile values
    sample_indices = [np.argmin(np.abs(predictions - value)) for value in percentile_values]

    for i, idx in enumerate(sample_indices):
        smiles = smiles_list[idx] if smiles_list is not None else f'Sample {idx}'
        plot_waterfall(explanation, idx, smiles, output_folder, percentiles[i])

    save_shap_values_to_csv(explanation, X, X.columns, output_folder)
    plot_summary_plots(explanation, output_folder)
    plot_scatter_plots(explanation, X.columns, output_folder)

    if smiles_list is not None and fingerprints is not None:
        for i, idx in enumerate(sample_indices):
            smiles = smiles_list[idx]
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                sample_shap_values = explanation[idx].values
                feature_names = X.columns
                valid_top_bits = []
                
                if fingerprints == 'morgan':
                    bit_info = {}
                    AllChem.GetHashedMorganFingerprint(mol, radius=RADIUS, nBits=NBITS, bitInfo=bit_info)
                    bit_shap_values = {}
                    for bit_idx, feature_name in enumerate(feature_names): 
                        bit = int(feature_name.split('-')[1])
                        if 0 <= bit_idx < len(sample_shap_values):
                            bit_shap_values[bit] = sample_shap_values[bit_idx]

                    valid_top_bits = [bit for bit in sorted(bit_shap_values.keys(), key=lambda b: abs(bit_shap_values[b]),
                                                            reverse=True) if bit in bit_info][:5]

                elif fingerprints == 'rdkit': 
                    bit_info = {}
                    Chem.RDKFingerprint(mol, minPath=_MIN_PATH_LEN, maxPath=_MAX_PATH_LEN, fpSize=NBITS, bitInfo=bit_info)
                    
                    bit_shap_values = {}
                    for bit_idx, feature_name in enumerate(feature_names): 
                        bit = int(feature_name.split('-')[1])
                        if 0 <= bit_idx < len(sample_shap_values):
                            bit_shap_values[bit] = sample_shap_values[bit_idx]

                    valid_top_bits = [bit for bit in sorted(bit_shap_values.keys(), key=lambda b: abs(bit_shap_values[b]),
                                                            reverse=True) if bit in bit_info][:5]

                draw_top_features(mol, bit_info, valid_top_bits, smiles,
                                        os.path.join(output_folder, f'sample_p{percentiles[i]}_top_features.png'), fingerprints)

                highlight_and_draw_molecule(mol, bit_info, valid_top_bits, bit_shap_values, smiles,
                                                os.path.join(output_folder,
                                                             f"sample_p{percentiles[i]}_shap_highlights.png"), fingerprints)

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


def draw_top_features(mol, bit_info, valid_top_bits, smiles, output_path, fingerprints):
    """Draw and save top features(bits)."""
    list_bits = []
    legends = []

    for x in valid_top_bits:
            for i in range(len(bit_info[x])):
                list_bits.append((mol, x, bit_info, i))
                legends.append(str(x))
    options = Draw.rdMolDraw2D.MolDrawOptions()
    options.prepareMolsBeforeDrawing = False    
    if fingerprints == 'morgan':
        p = Draw.DrawMorganBits(list_bits, molsPerRow=6, legends=legends, drawOptions=options)
        p.save(output_path)
    elif fingerprints == 'rdkit':
        p = Draw.DrawRDKitBits(list_bits, molsPerRow=6, legends=legends, drawOptions=options)
        p.save(output_path)

    add_title_to_image(output_path, f"Top 5 features({fingerprints}-fps) for: {smiles}")

def highlight_and_draw_molecule(mol, bit_info, valid_top_bits, bit_shap_values, smiles, output_path, fingerprints):
    highlights, atom_colors, bond_highlights, bond_colors = set(), {}, set(), {}

    def add_highlights(atoms, bonds, color):
        highlights.update(atoms)
        bond_highlights.update(bonds)
        for atom in atoms:
            atom_colors[atom] = color
        for bond in bonds:
            bond_colors[bond] = color
    # print('smiles', smiles, 'bits: ', valid_top_bits)
    for bit in valid_top_bits:
        shap_value = bit_shap_values.get(bit, 0)
        color = (1, 0, 0, 0.6) if shap_value > 0 else (0, 0, 1, 0.6)
        bit_atoms = bit_info.get(bit)

        if fingerprints == 'morgan':
            atom_indices = [item for sublist in bit_atoms for item in sublist]
            for atom_idx in atom_indices:
                if isinstance(atom_idx, int):
                    env = Chem.FindAtomEnvironmentOfRadiusN(mol, RADIUS, atom_idx)
                    atoms, bonds = set(), set()
                    for bidx in env:
                        bond = mol.GetBondWithIdx(bidx)
                        atoms.update([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
                        bonds.add(bidx)
                    add_highlights(atoms, bonds, color)
                        
        elif fingerprints == 'rdkit':
            for env in bit_atoms:  
                atoms, bonds = set(), set()                    
                for idx in env:
                    atoms.add(mol.GetBondWithIdx(idx).GetBeginAtomIdx())
                    atoms.add(mol.GetBondWithIdx(idx).GetEndAtomIdx())
                    bonds.add(idx) 
                add_highlights(atoms, bonds, color)

    drawer = MolDraw2DCairo(500, 500)
    drawer.drawOptions().useBWAtomPalette()
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, mol, highlightAtoms=list(highlights), 
        highlightAtomColors=atom_colors, highlightBonds=list(bond_highlights), 
        highlightBondColors=bond_colors
    )
    drawer.FinishDrawing()
    drawer.WriteDrawingText(output_path)
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
    