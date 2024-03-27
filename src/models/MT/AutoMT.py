import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class AutoMT(nn.Module):
    def __init__(self, model, device):
        super(AutoMT, self).__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model).to(self.device)

    def forward(self, inputs):
        model_inputs = self.tokenizer(inputs, return_tensors="pt", padding=True).to(self.device)
        generated_tokens = self.model.generate(**model_inputs).to(self.device)
        translations = self.tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )
        return translations
