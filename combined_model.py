import torch
import torch.nn as nn
import torch.nn.functional as F

class DrugNet(nn.Module):
    def __init__(self, gnn, transformer, hidden_size=256):
        super(DrugNet, self).__init__()
        self.gnn = gnn
        self.transformer = transformer
        self.fc1 = nn.Linear(gnn.classify.out_features + transformer.transformer.config.hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, g, features, smiles_list):
        gnn_out = self.gnn(g, features)
        trf_out = self.transformer(smiles_list)
        combined = torch.cat([gnn_out, trf_out], dim=1)
        x = F.relu(self.fc1(combined))
        return torch.sigmoid(self.fc2(x))
