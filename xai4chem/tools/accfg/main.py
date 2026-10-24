import csv
import os
from importlib import resources

from rdkit import Chem

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

def canonical_smiles(smi):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smi))

class AccFG:
    def __init__(self, common_fgs=True, heterocycle_fgs=True, user_defined_fgs={}, print_load_info=False, lite=False):
        self.lite = lite
        log_text = ""
        if common_fgs and not lite:
            self.dict_fgs_common_path = resources.files("xai4chem").joinpath("tools/accfg/fgs_common.csv")

            self.dict_fgs_common = self.csv_to_dict(self.dict_fgs_common_path)
            log_text += f"Loaded {len(self.dict_fgs_common)} common functional groups. "
        elif common_fgs and lite:
            self.dict_fgs_common_path = resources.files("xai4chem").joinpath("tools/accfg/fgs_common.csv")

            self.dict_fgs_common = self.csv_to_dict(self.dict_fgs_common_path, lite=self.lite)
            log_text += f"Loaded {len(self.dict_fgs_common)} common functional groups (lite). "
        else:
            self.dict_fgs_common = {}
        if heterocycle_fgs:
            self.dict_fg_heterocycle_path =  resources.files("xai4chem").joinpath("tools/accfg/fgs_heterocycle.csv")

            self.dict_fg_heterocycle = self.csv_to_dict(self.dict_fg_heterocycle_path)
            log_text += f"Loaded {len(self.dict_fg_heterocycle)} heterocycle groups. "
        else:
            self.dict_fg_heterocycle = {}
        if user_defined_fgs:
            self.dict_fgs_user_defined = self.process_user_defined_fgs(user_defined_fgs)
            log_text += f"Loaded {len(user_defined_fgs)} user-defined functional groups. "
        else:
            self.dict_fgs_user_defined = {}
        self.dict_fgs = {**self.dict_fgs_common, **self.dict_fg_heterocycle, **self.dict_fgs_user_defined}

        if print_load_info:
            print(f'{log_text}Total {len(self.dict_fgs)} functional groups loaded.')

    def _is_fg_in_mol(self, mol, fg):
        fgmol = Chem.MolFromSmarts(fg)
        # mol = Chem.MolFromSmiles(mol.strip())
        mapped_atoms = Chem.Mol.GetSubstructMatches(mol, fgmol, uniquify=True)
        if_mapped = len(mapped_atoms) > 0
        return if_mapped, mapped_atoms

    def _freq_fg_in_mol(self, mol, fg):
        fgmol = Chem.MolFromSmarts(fg)
        # mol = Chem.MolFromSmiles(mol.strip())
        freq = len(Chem.Mol.GetSubstructMatches(mol, fgmol, uniquify=True))
        if freq > 0:
            return freq
        else:
            return False
    def _get_bonds_from_match(self, query_mol, mol, atom_match):
        """
        Args:
        query_mol: query molecule used to match
        mol: molecule matched
        atom_match: result of GetSubstructMatch (i.e. matched atom idx list)
        Returns:
        list of matched bond indices, or None
        """
        bonds = []
        if isinstance(atom_match, (list, tuple)):
            pass
        else:
            atom_match = list(atom_match)
        for bond in query_mol.GetBonds():
            idx1, idx2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bonds.append(mol.GetBondBetweenAtoms(atom_match[idx1], atom_match[idx2]).GetIdx())
        return bonds

    def process_user_defined_fgs(self, user_defined_fgs):
        user_defined_fgs_edit = {}
        for fg_name, fg_smi in user_defined_fgs.items():
            if ';' in fg_smi:
                fg_smi_edit = fg_smi
            else:
                fg_smi_edit = Chem.MolToSmiles(Chem.MolFromSmiles(fg_smi))
                fg_smi_edit = fg_smi_edit.replace('[nH]','[n]')
            user_defined_fgs_edit[fg_name] = fg_smi_edit
        return user_defined_fgs_edit

    def run(self, smiles: str, show_atoms=True, show_graph=False, canonical=True) -> dict:
        """
        Input a molecule SMILES or name.
        Returns a list of functional groups identified by their common name (in natural language).
        """
        if canonical:
            smiles = canonical_smiles(smiles)
        mol = Chem.MolFromSmiles(smiles)
        return self.run_mol(mol, show_atoms=show_atoms, show_graph=show_graph)

    def run_mol(self, mol: object, show_atoms=True, show_graph=False, use_atom_map_num=False):
        fgs_in_molec = {name: self._is_fg_in_mol(mol, fg)[1] for name, fg in self.dict_fgs.items() if self._is_fg_in_mol(mol, fg)[0]}
            # Check if the functional groups are subgroups of other functional groups
            # Build FG graph

        if use_atom_map_num:
            atom_idx_map_num_dict = {}
            for atom in mol.GetAtoms():
                atom_idx_map_num_dict[atom.GetIdx()] = atom.GetAtomMapNum()
            fgs_in_molec = {fg: [[atom_idx_map_num_dict[atom] for atom in mapped_atoms] for mapped_atoms in mapped_atoms_list] for fg, mapped_atoms_list in fgs_in_molec.items()}
        if show_atoms and not show_graph:
            return fgs_in_molec
        elif show_atoms and show_graph:
            return fgs_in_molec, fg_graph
        else:
            return list(fgs_in_molec.keys())

    def run_freq(self, smiles: str) -> str:
        """
        Input a molecule SMILES or name.
        Returns a list of functional groups identified by their common name (in natural language).
        """
        try:
            fgs = self.run(smiles, show_atoms=True)
            fgs_freq = [(fg, len(mapped_atoms)) for fg, mapped_atoms in fgs.items()]
            return fgs_freq
        except:
            return None
            # return "Wrong argument. Please input a valid molecular SMILES."
    def csv_to_dict(self, csv_file, lite=False):
        data = {}
        with open(csv_file, 'r') as file:
            if not lite:
                reader = csv.DictReader(filter(lambda row: not row.lstrip().startswith('%'), file))
            else:
                reader = csv.DictReader(filter(lambda row: not row.lstrip().startswith(('#','%')), file))
            for row in reader:
                key = row.pop('Functional Group')
                if not lite and key.lstrip().startswith('#'):
                    key = key.replace('#','').lstrip()
                data[key] = row.pop('SMARTS Pattern')
        return data
