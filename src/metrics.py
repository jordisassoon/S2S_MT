import evaluate
from transformers import pipeline

# Utility functions
def compute_metrics(outputs, predictions, device):
    # Compute all the metrics based on the outputs and the predicted outputs
    
    # Returns transcripts of predicted outputs
    pip = pipeline("automatic-speech-recognition", model="openai/whisper-base", tokenizer="openai/whisper-base", device=device)
    # For now we translate from english to french
    pred = [pip(predictions, generate_kwargs = {"task":"transcribe", "language":"<|fr|>"} )['text']]

    print("Real translation :", outputs)
    print("Predicted translation :", pred)

    # BLEU and charBLEU scores
    bleu, charbleu = compute_BLEU(outputs, pred)

    # chrF
    chrf = compute_chrf(outputs, pred)

    # MCD
    mcd = compute_MCD(outputs, pred)

    return bleu, charbleu, chrf, mcd

# Metrics on transcripts
def compute_BLEU(outputs, predictions):
    # computes BLEU and charBLEU
    sacrebleu = evaluate.load("sacrebleu")
    bleu = sacrebleu.compute(predictions=predictions, references=outputs, lowercase = True)

    charbleu = sacrebleu.compute(predictions=predictions, references=outputs, tokenize = "char", lowercase = True)
    return bleu['score'], charbleu['score']
    
def compute_chrf(outputs, predictions):
    # computes chrF
    chrf = evaluate.load("chrf")
    chrf_score = chrf.compute(predictions=predictions, references=outputs, lowercase = True)
    return chrf_score['score']
    
# Metrics on audios
def compute_MCD(outputs, predictions):
    # computes MCD

    return