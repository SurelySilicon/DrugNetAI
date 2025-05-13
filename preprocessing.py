# utils/preprocessing.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from rdkit import Chem

def clean_smiles(smiles_list):
    """Remove invalid SMILES and return cleaned list."""
    cleaned = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            cleaned.append(Chem.MolToSmiles(mol))
    return cleaned

def scale_features(df, cols_to_scale):
    """Standardize selected numerical columns."""
    scaler = StandardScaler()
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    return df
