import evaluate
from transformers import pipeline

import librosa
from dtw import dtw
import numpy as np


# Utility functions
def compute_metrics(outputs, predictions, outputs_files, predictions_files, device):
    # Compute all the metrics based on the outputs and the predicted outputs

    # Returns transcripts of predicted outputs
    pip = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-base",
        tokenizer="openai/whisper-base",
        device=device,
    )
    # For now we translate from english to french
    pred = [
        item["text"]
        for item in pip(
            predictions, generate_kwargs={"task": "transcribe", "language": "<|fr|>"}
        )
    ]

    print("Real translation :", outputs)
    print("Predicted translation :", pred)

    # BLEU and charBLEU scores
    bleu, charbleu = compute_BLEU(outputs, pred)

    # chrF
    chrf = compute_chrf(outputs, pred)

    # MCD
    mcd = compute_MCD(outputs_files, predictions_files)

    return bleu, charbleu, chrf, mcd


# Metrics on transcripts
def compute_BLEU(outputs, predictions):
    # Computes BLEU and charBLEU
    sacrebleu = evaluate.load("sacrebleu")
    bleu = sacrebleu.compute(
        predictions=predictions, references=outputs, lowercase=True
    )

    charbleu = sacrebleu.compute(
        predictions=predictions, references=outputs, tokenize="char", lowercase=True
    )
    return bleu["score"], charbleu["score"]


def compute_chrf(outputs, predictions):
    # Computes chrF
    chrf = evaluate.load("chrf")
    chrf_score = chrf.compute(
        predictions=predictions, references=outputs, lowercase=True
    )
    return chrf_score["score"]


# Metrics on audios
def compute_MCD(outputs, predictions):
    # Load audios
    sr = 16000
    original_audio, _ = librosa.load(outputs, sr=sr)
    synthesized_audio, _ = librosa.load(predictions, sr=sr)

    # Extract MFCC features
    original_mfcc = librosa.feature.mfcc(y = original_audio, sr=sr)
    synthesized_mfcc = librosa.feature.mfcc(y = synthesized_audio, sr=sr)

    # Align the MFCC features using DTW
    _, _, _, path = dtw(original_mfcc.T, synthesized_mfcc.T, dist=lambda x, y: np.linalg.norm(x - y, ord=1))

    # Select aligned frames
    aligned_original_mfcc = original_mfcc[:, path[0]]
    aligned_synthesized_mfcc = synthesized_mfcc[:, path[1]]

    # Compute the Euclidean distance between aligned frames
    distances = np.sqrt(2 * np.sum((aligned_original_mfcc - aligned_synthesized_mfcc) ** 2, axis=0))

    # Compute the mean distance
    mean_distance = np.mean(distances)

    # Compute MCD
    mcd = (10 / np.log(10)) * mean_distance

    return mcd
