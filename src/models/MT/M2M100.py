import torch.nn as nn
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer


class M2M100(nn.Module):
    def __init__(self, model, src_lang, tgt_lan, device):
        super(M2M100, self).__init__()
        self.tokenizer = M2M100Tokenizer.from_pretrained(model)
        self.model = M2M100ForConditionalGeneration.from_pretrained(model).to(device)
        self.tgt_lan = tgt_lan

    def forward(self, inputs):
        model_inputs = self.tokenizer(inputs, return_tensors="pt")
        generated_tokens = self.model.generate(
            **model_inputs, forced_bos_token_id=self.tokenizer.get_lang_id(self.tgt_lan)
        )
        translations = self.tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )
        return translations
