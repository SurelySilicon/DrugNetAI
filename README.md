🧬 DrugNetAI
A powerful deep learning system that combines Graph Neural Networks (GNNs) and Transformer encoders to predict the drug-likeness of chemical molecules using their SMILES (Simplified Molecular Input Line Entry System) representations.

Created and maintained by: @SurelySilicon

🚀 Project Overview
DrugNetAI is a cutting-edge AI model that:

Parses chemical structures using SMILES strings

Converts them into molecular graphs (atoms as nodes, bonds as edges)

Processes them through a Graph Neural Network (GNN)

Encodes the same SMILES using a Transformer

Fuses both modalities to classify whether the molecule is a potential drug candidate

🧠 Model Architecture
Input: SMILES string (e.g., "CCO" for ethanol)
Branch 1: Graph Neural Network using DGL
Branch 2: Transformer encoder using HuggingFace's pretrained SMILES-BERT
Fusion: Concatenated feature vector → Fully connected layers
Output: Binary classification (drug-like or not)

📁 Project Structure
DrugNetAI/
├── gnn_model.py → GNN for molecular graphs
├── transformer_encoder.py → HuggingFace transformer for SMILES
├── combined_model.py → Final model merging GNN + Transformer
├── mol_utils.py → SMILES to RDKit + DGL graph conversion
├── train.py → Model training script
├── predict.py → Run predictions on new molecules
├── utils/
│ └── preprocessing.py → New! Clean SMILES & scale features for model input ✅
├── requirements.txt → All dependencies
└── README.md → This file

🔧 Installation
To install dependencies:
pip install -r requirements.txt

If you're using RDKit (recommended), install it via Conda:
conda install -c rdkit rdkit

📊 Training the Model
To train the model:
python train.py

Hyperparameters can be manually configured inside train.py.

🔍 Making Predictions
To predict drug-likeness for a new molecule:
python predict.py "CCO"
(This example uses the SMILES for ethanol)

🔧 Utilities (New)
utils/preprocessing.py includes reusable preprocessing tools:

clean_smiles: Removes invalid SMILES and standardizes valid ones using RDKit

scale_features: Scales numerical features using Scikit-learn’s StandardScaler

These functions ensure that your data is clean, chemically valid, and machine-learning ready.

📦 Dependencies
PyTorch

DGL (Deep Graph Library)

HuggingFace Transformers

RDKit

Scikit-learn

Pandas / NumPy

🧪 Sample SMILES Examples
Molecule: Ethanol → SMILES: CCO
Molecule: Aspirin → SMILES: CC(=O)OC1=CC=CC=C1C(=O)O
Molecule: Benzene → SMILES: C1=CC=CC=C1

📌 Motivation
DrugNetAI explores how graph-based molecular representations and language-based SMILES encodings can be jointly leveraged through modern deep learning architectures to advance drug discovery pipelines.

✨ Author
SurelySilicon

