import inspect
import torch
import torch.nn.functional as F
torch.cuda.empty_cache()
import pathlib
import os
from discrete_diffusion_scheduler import DiffusionTransformer
from diffusers.training_utils import EMAModel
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from models.model import D2GraphTransformer
from datasets.zinc250k_dataset import Zin250KDataset
import argparse
import rdkit
from rdkit import Chem
import re
import networkx as nx


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", type=str, default='data/zinc250k/')
parser.add_argument("--add-position", type=bool, default=True)
parser.add_argument("--hidden-size", type=int, default=128)
parser.add_argument("--num-layers", type=int, default=6)
parser.add_argument("--num-heads", type=int, default=8)
parser.add_argument("--drop-rate", type=float, default=0.0)
parser.add_argument("--last-drop-rate", type=float, default=0.0)
parser.add_argument("--batch-size", type=int, default=128)

parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--diffusion_steps", type=int, default=500)

parser.add_argument("--sample-num", type=int, default=10000)


args = parser.parse_args()

train_dataset = Zin250KDataset(stage='train',
                           root=args.data_dir)
test_dataset = Zin250KDataset(stage='test',
                          root=args.data_dir)
valid_dataset = Zin250KDataset(stage='val',
                           root=args.data_dir)

charge_decoder = {j:i for i,j in train_dataset.charges.items()}
bond_decoder = {j:i for i,j in train_dataset.bonds.items()}
atom_decoder = {j:i for i,j in train_dataset.types.items()}

def mol2smiles(mol):
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return None

    return Chem.MolToSmiles(mol,canonical=True)

def build_molecule(atom_types,atom_charges, edge_types, verbose=True):
    if verbose:
        print("building new molecule")
    atom_index = []
    for i,atom in enumerate(atom_types):
        if atom != 0 and atom !=1:
            atom_index.append(i)
    atom_types = atom_types[atom_index]
    atom_charges = atom_charges[atom_index]
    edge_types = edge_types[atom_index,:][:,atom_index]

    A = nx.from_numpy_array((edge_types>1).float().numpy())
    cl = nx.connected_components(A)
    atom_index = list(max(cl,key=len))
    atom_types = atom_types[atom_index]
    atom_charges = atom_charges[atom_index]
    edge_types = edge_types[atom_index,:][:,atom_index]

    mol = Chem.RWMol()

    for i,atom in enumerate(atom_types):
        a = Chem.Atom(atom_decoder[atom.item()])
        a.SetFormalCharge(charge_decoder[atom_charges[i].item()])
        mol.AddAtom(a)
        if verbose:
            print("Atom added: ", atom.item(), atom_decoder[atom.item()])

    edge_types = torch.tril(edge_types)
    all_bonds = torch.nonzero((edge_types>1).float())
    for i, bond in enumerate(all_bonds):
        if bond[0].item() != bond[1].item():
            if  edge_types[bond[0], bond[1]].item() ==0:
                continue
            mol.AddBond(bond[0].item(), bond[1].item(), bond_decoder[edge_types[bond[0], bond[1]].item()])
            if verbose:
                print("bond added:", bond[0].item(), bond[1].item(), edge_types[bond[0], bond[1]].item(),
                      bond_decoder[edge_types[bond[0], bond[1]].item()] )
    return mol
def main():
    print(args)

    model = D2GraphTransformer(atom_dim=len(train_dataset.types),charge_dim=len(train_dataset.charges),edge_dim=len(train_dataset.bonds),
                 num_layers=args.num_layers,d_model=args.hidden_size,num_heads=args.num_heads,
                               dff=args.hidden_size,position=args.add_position)
    model = model.to(device=device)

    noise_scheduler = DiffusionTransformer(mask_id=0 ,a_classes=len(train_dataset.types),c_classes=len(train_dataset.charges),
                e_classes=len(train_dataset.bonds),model=model,diffusion_step=500,max_length=train_dataset.max_length)
    noise_scheduler = noise_scheduler.to(device)

    p = '_position' if args.add_position else ''
    noise_scheduler.model.load_state_dict(torch.load(f'zinc250k_weights{p}/model_weights_{args.epochs}'
                                                        f'.pt',map_location=device))

    smiles_list = []
    model.eval()
    t = 0
    with torch.no_grad():
        molecule_list = noise_scheduler.sample(args.sample_num)
        for molecule in molecule_list:
            A,C,E = molecule
            t = t + A.shape[0]

            for i in range(A.shape[0]):
                atom_types = A[i].cpu()
                edge_types = E[i].cpu()
                atom_charges = C[i].cpu()

                mol = build_molecule(atom_types, atom_charges, edge_types,verbose=False)
                smiles = mol2smiles(mol)
                if smiles is not None:
                    smiles_list.append(smiles)
                else:
                    print(smiles)

    df = train_dataset.data
    train_smiles_list = df['CAN_SMILES'].to_list()
    unique_list = list(set(smiles_list))
    novelty_list = []
    for smiles in set(unique_list):
        if smiles not in train_smiles_list:
            novelty_list.append(smiles)

    print('valid:',len(smiles_list)/t)
    print('unique:', len(set(smiles_list)) / len(smiles_list))
    print('novelty:',len(novelty_list)/len(unique_list))
    with open(args.data_dir + f'/results_{args.epochs}.txt','wt') as f:
        for smiles in smiles_list:
            f.write(smiles + '\n')


if __name__ == "__main__":
    main()
