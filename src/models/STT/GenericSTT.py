import torch
import torch.nn as nn
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration


class GenericSTT(nn.Module):
    def __init__(self, model, device):
        super(GenericSTT, self).__init__()
        self.device = device
        self.processor = Speech2TextProcessor.from_pretrained(model)
        self.model = Speech2TextForConditionalGeneration.from_pretrained(model).to(
            self.device
        )

    def forward(self, inputs, sampling_rate):
        input_features = self.processor(
            inputs, sampling_rate=sampling_rate, return_tensors="pt", padding=True
        ).input_features.to(self.device)
        generated_ids = self.model.generate(inputs=input_features)
        transcriptions = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        return transcriptions
