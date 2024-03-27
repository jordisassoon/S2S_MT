import torch
import torch.nn as nn
from transformers import VitsModel, AutoTokenizer


class VITS(nn.Module):
    def __init__(self, model, device):
        super(VITS, self).__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = VitsModel.from_pretrained(model).to(self.device)

    def forward(self, inputs):
        input_features = self.tokenizer(inputs, return_tensors="pt", padding=True).to(self.device)
        output = self.model(**input_features).waveform.to(self.device)
        return output
