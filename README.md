# 🧬 DrugNetAI

A powerful deep learning system that combines **Graph Neural Networks (GNNs)** and **Transformer encoders** to predict the drug-likeness of chemical molecules using their **SMILES (Simplified Molecular Input Line Entry System)** representations.

Created and maintained by [@SurelySilicon](https://github.com/SurelySilicon).

---

## 🚀 Project Overview

DrugNetAI is a cutting-edge AI model that:
- Parses chemical structures using **SMILES**
- Converts them into **molecular graphs** (atoms as nodes, bonds as edges)
- Processes them through a **Graph Neural Network (GNN)**
- Encodes the same SMILES using a **Transformer**
- Fuses both modalities to classify whether the molecule is a potential drug

---

## 🧠 Model Architecture

- **Input**: SMILES string (e.g., `"CCO"` for ethanol)
- **Branch 1**: Graph Neural Network using DGL (Deep Graph Library)
- **Branch 2**: Transformer encoder using HuggingFace (pretrained SMILES-BERT)
- **Fusion**: Concatenated vector → fully connected layers → output
- **Output**: Binary classification (drug-like or not)

---

## 📁 Project Structure

```bash
DrugNetAI/
├── gnn_model.py             # GNN for molecular graphs
├── transformer_encoder.py   # HuggingFace transformer for SMILES
├── combined_model.py        # Final model merging GNN + Transformer
├── mol_utils.py             # SMILES to RDKit + DGL graph conversion
├── train.py                 # Model training script
├── predict.py               # Run predictions on new molecules
├── requirements.txt         # All dependencies
└── README.md                # This file
```

🔧 Installation

pip install -r requirements.txt
Make sure to also install RDKit manually if needed:


conda install -c rdkit rdkit

📊 Training
Train the model using:
python train.py
You can configure hyperparameters in the script itself.

🔍 Prediction
Predict drug-likeness for new SMILES strings:
python predict.py "CCO"  # Example SMILES for ethanol

📦 Dependencies
PyTorch

DGL

Transformers (HuggingFace)

RDKit

Scikit-learn

Pandas / NumPy

🧪 Sample SMILES
Molecule	SMILES
Ethanol	CCO
Aspirin	CC(=O)OC1=CC=CC=C1C(=O)O
Benzene	C1=CC=CC=C1

📌 Motivation
This project explores how graph-based molecular representations and language-based SMILES encodings can be jointly leveraged to improve drug discovery pipelines using modern deep learning architectures.

✨ Author
SurelySilicon
