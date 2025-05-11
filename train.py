import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

from gnn_model import GNN
from transformer_encoder import SMILESEncoder
from combined_model import DrugNet
from mol_utils import smiles_to_graph

# Sample dataset (replace with your real one from Kaggle or any Medical Dataset Site :) I used my Private one so :)
data = {
    'smiles': ['CCO', 'CC(=O)OC1=CC=CC=C1C(=O)O', 'C1=CC=CC=C1'],  # Ethanol, Aspirin, Benzene
    'label': [1, 1, 0]  # 1 = drug-like, 0 = not
}
df = pd.DataFrame(data)

# Split dataset
train_smiles, test_smiles, y_train, y_test = train_test_split(
    df['smiles'], df['label'], test_size=0.3, random_state=42
)

# Model setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gnn = GNN(in_feats=1, hidden_size=64, num_classes=64)
transformer = SMILESEncoder()
model = DrugNet(gnn, transformer).to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
model.train()
for epoch in range(10):
    total_loss = 0
    for smiles, label in zip(train_smiles, y_train):
        try:
            g, features = smiles_to_graph(smiles)
        except ValueError:
            continue
        g = g.to(device)
        features = features.to(device)
        label = torch.tensor([[label]], dtype=torch.float32).to(device)

        optimizer.zero_grad()
        output = model(g, features, [smiles])
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
