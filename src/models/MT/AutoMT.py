import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class AutoMT(nn.Module):
    def __init__(self, model, device):
        super(AutoMT, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model).to(device)

    def forward(self, inputs):
        model_inputs = self.tokenizer(inputs, return_tensors="pt")
        generated_tokens = self.model.generate(**model_inputs)
        translations = self.tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )
        return translations
