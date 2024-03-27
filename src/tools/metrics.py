import evaluate
from transformers import pipeline

import librosa
from dtw import dtw
import numpy as np


# Utility functions
def compute_metrics(target_text, target_audio, translated_audio, device):
    # Compute all the metrics based on the outputs and the predicted outputs

    # Returns transcripts of predicted outputs
    pip = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-base",
        tokenizer="openai/whisper-base",
        device=device,
    )
    # For now we translate from english to french
    transcribed_translated_audio = [item['text'] for item in pip(
            translated_audio, generate_kwargs={"task": "transcribe", "language": "<|fr|>"}
        )]

    # BLEU, charBLEU, chrF
    bleu, charbleu, chrf = compute_all_text_metrics(target_text, transcribed_translated_audio)

    # MCD
    mcd = 0
    for i in range(len(target_audio)):
        mcd += compute_MCD(np.array(target_audio[i]), translated_audio[i])

    return bleu, charbleu, chrf, mcd/len(target_audio), transcribed_translated_audio


# Metrics on transcripts
def compute_all_text_metrics(target_text, transcribed_translated_audio):
    # BLEU and charBLEU scores
    bleu, charbleu = compute_BLEU(target_text, transcribed_translated_audio)

    # chrF
    chrf = compute_chrf(target_text, transcribed_translated_audio)

    return bleu, charbleu, chrf

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
def compute_MCD(original_audio, synthesized_audio):
    # Model's sr
    sr = 16000

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
