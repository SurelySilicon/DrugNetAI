import torch
from gnn_model import GNN
from transformer_encoder import SMILESEncoder
from combined_model import DrugNet
from mol_utils import smiles_to_graph

# Load models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gnn = GNN(in_feats=1, hidden_size=64, num_classes=64)
transformer = SMILESEncoder()
model = DrugNet(gnn, transformer).to(device)

# Load saved weights if available (optional)
# model.load_state_dict(torch.load('drugnet_model.pth'))

model.eval()

def predict_smiles(smiles):
    try:
        g, features = smiles_to_graph(smiles)
        g = g.to(device)
        features = features.to(device)
        with torch.no_grad():
            output = model(g, features, [smiles])
            prediction = output.item()
            print(f"SMILES: {smiles} → Prediction: {prediction:.4f}")
            return prediction
    except Exception as e:
        print(f"Error processing {smiles}: {e}")
        return None

# Example usage
test_smiles_list = ["CCO", "C1=CC=CC=C1", "CCN(CC)CC"]
for sm in test_smiles_list:
    predict_smiles(sm)
