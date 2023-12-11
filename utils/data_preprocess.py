import torch
import rdkit
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Data: 
    def __init__(self, smi_path, coor_path) :
        self.smi_path = smi_path
        self.coor_path = coor_path 
        self.coor_list = self.get_coor()
        self.smi_list = self.get_smi()

        self.smi_dic = self.get_dic()

        self.dic_len = len(self.smi_dic)
        self.longest_smi = self.get_longest(self.smi_list)
        self.longest_coor = self.get_longest(self.coor_list)

        self.smint_list = self.get_smint()


    def get_coor(self) :
        all_coordinate = []
        supplier = rdkit.Chem.SDMolSupplier(self.coor_path)
        for mol in supplier:
            coor_list = []
            if mol is not None:
                conformer = mol.GetConformer()
                for atom in mol.GetAtoms():
                    atom_idx = atom.GetIdx()
                    x, y, z = conformer.GetAtomPosition(atom_idx)
                    coor_atom = list((x,y,z))
                    coor_list.append(coor_atom)
            all_coordinate.append(coor_list)

        return all_coordinate



    def get_smi(self) :
        smi_df = pd.read_csv(self.smi_path)
        smi_list = list(smi_df['Canonical SMILES'])
        smi_list = [smi + 'E' for smi in smi_list]

        smi_list = self.replace_duplicate_atom(smi_list) 
        return smi_list



    def get_dic(self) :
        smi_dic = {'x': 0, 'E': 1}
        i = len(smi_dic)
        for smi in self.smi_list :
                for atom in smi :
                    if atom not in smi_dic :
                        smi_dic[atom] = i
                        i += 1
        return smi_dic



    def get_longest(self, input_list) :
        longest = 0
        for i in input_list :
            if len(i) > longest :
                longest = len(i)
        return longest






    def get_smint(self) :
        smint_list = []
        for smi in self.smi_list :
            smi = list(smi)
            smint = [self.smi_dic[atom] for atom in smi]
            smint = smint + [0] * (self.longest_smi - len(smint)) # Pad with 0
            smint_list.append(smint)

        return smint_list



    def replace_duplicate_atom(self, smi_list) :
        smi_list = [smi.replace('X', 'Na')
                        .replace('Y', 'Cl')
                        .replace('Z', 'Br')
                        .replace('T', 'Ba') for smi in smi_list]

        return smi_list




    def extract(self) :
        print(f"Train data extracted\nSize: {len(self.smint_list)}\nLongest SMILES: {self.longest_smi}\nLongest Coordinate: {self.longest_coor}")
        print("----------------------------------------")
        print(f"Sample x: {self.smint_list[0]}")
        print(f"Sample y: {self.coor_list[0]}")
        return self.smint_list, self.coor_list, self.smi_list, self.smi_dic, self.longest_smi, self.longest_coor



class TorchDataset(Dataset) :
  def __init__(self, x, y) :
    self.x = x
    self.y = y

  def __len__(self) :
    return len(self.x)

  def __getitem__(self, idx) :
    return torch.tensor(self.x[idx], dtype = torch.long, device=device), torch.tensor(self.y[idx], device = device)
  


def normalize_coor(coor_list) :
    n_coor_list = []

    for mol_coor in coor_list :
        n_mol_coor = []

        x_origin, y_origin, z_origin = mol_coor[0]

        for atom_coor in mol_coor :
            n_atom_coor = [round(atom_coor[0] - x_origin, 2), 
                        round(atom_coor[1] - y_origin, 2), 
                        round(atom_coor[2] - z_origin, 2)]
            n_mol_coor.append(n_atom_coor)
        n_coor_list.append(n_mol_coor)
    return n_coor_list
          

def pad_coor(coor_list, longest_coor) :
    p_coor_list = []

    for i in coor_list :
        if len(i) < longest_coor :
            zeros = [[0,0,0]] * (longest_coor - len(i))
            zeros = torch.tensor(zeros)
            i = torch.tensor(i)
            i = torch.cat((i, zeros), dim = 0)
            p_coor_list.append(i)
        else :
            p_coor_list.append(i)
    return p_coor_list