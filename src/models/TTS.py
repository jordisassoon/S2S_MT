import torch.nn as nn
from transformers import VitsModel, AutoTokenizer


class TextToSpeech(nn.Module):
    def __init__(self, model, device):
        super(TextToSpeech, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = VitsModel.from_pretrained(model).to(device)

    def forward(self, inputs):
        input_features = self.tokenizer(inputs, return_tensors="pt")
        output = self.model(**input_features).waveform
        return output
