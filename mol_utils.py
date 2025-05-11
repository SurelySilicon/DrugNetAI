from rdkit import Chem
from rdkit.Chem import AllChem
import dgl
import torch

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    atoms = mol.GetAtoms()
    g = dgl.DGLGraph()
    g.add_nodes(len(atoms))

    node_feats = []
    for atom in atoms:
        node_feats.append([atom.GetAtomicNum()])  # Use atomic number here bro :)

    for bond in mol.GetBonds():
        g.add_edges(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        g.add_edges(bond.GetEndAtomIdx(), bond.GetBeginAtomIdx())

    g.ndata['feat'] = torch.tensor(node_feats).float()
    return g, g.ndata['feat']
