import torch.nn as nn
from transformers import WhisperProcessor, WhisperForConditionalGeneration

class WhisperSTT(nn.Module):
    def __init__(self, model, device):
        super(WhisperSTT, self).__init__()
        self.device = device
        self.processor = WhisperProcessor.from_pretrained(model)
        self.model = WhisperForConditionalGeneration.from_pretrained(model).to(self.device)

    def forward(self, inputs, sampling_rate):
        input_features = self.processor(
            inputs, sampling_rate=sampling_rate, return_tensors="pt", padding=True
        ).input_features.to(self.device)
        generated_ids = self.model.generate(inputs=input_features).to(self.device)
        transcriptions = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        return transcriptions
