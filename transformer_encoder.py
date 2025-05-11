from transformers import AutoTokenizer, AutoModel
import torch.nn as nn

class SMILESEncoder(nn.Module):
    def __init__(self, model_name="seyonec/ChemBERTa-zinc-base-v1"):
        super(SMILESEncoder, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
    
    def forward(self, smiles_list):
        encoded_input = self.tokenizer(smiles_list, return_tensors='pt', padding=True, truncation=True)
        output = self.transformer(**encoded_input)
        pooled = output.last_hidden_state.mean(dim=1)
        return pooled
