import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Utility functions
def transcripts_audio(outputs, predicitions):
    # Returns transcripts of outputs and predicted outputs
    processor = WhisperProcessor.from_pretrained("openai/whisper-base")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
    # For now we translate to english
    model.config.forced_decoder_ids = WhisperProcessor.get_decoder_prompt_ids(language="english", task="transcribe")

    input_features = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_features

    return

# Metrics on transcripts
def compute_BLEU(outputs, predicitions):
    # TO DO

    return
    
def compute_chrf(outputs, predicitions):
    # TO DO

    return
    
def compute_charBLEU(outputs, predicitions):
    # TO DO

    return
    
# Metrics on audios
def compute_MCD(outputs, predicitions):
    # TO DO

    return