import argparse
from datasets import load_dataset

from models.STT.GenericSTT import GenericSTT
from models.MT.M2M100 import M2M100
from models.MT.AutoMT import AutoMT
from models.TTS.VITS import VITS

from utils.metrics import compute_metrics
from utils.save import save_wav

from tqdm import tqdm
import librosa
import torch
from torch.utils.data import DataLoader
import numpy as np

from token_hf import token


def sample_audio(audio):
    speech_array, _ = librosa.load(audio["file"], sr=16000) # model's sampling rate
    return speech_array


def process_batches(S2T, MT, T2S, sampling_rate, args, batch_size=5):
    # Load datasets
    # Target, English
    cvss = load_dataset(
        "google/cvss", "cvss_c", languages=["fr"], split="validation", trust_remote_code=True
    )
    
    # Source, French
    common_voice = load_dataset(
        "mozilla-foundation/common_voice_4_0", "fr", split="validation", trust_remote_code=True, streaming=True, token=token
    )

    if len(cvss) < batch_size:
        batch_size = len(cvss)

    # TODO:
    # Scores over all the instances

    target = iter(common_voice)
    count = 0
    for b in tqdm(range(len(cvss))):
        # Iterate through Common Voice
        s = next(target, 'END')
        if s == 'END':
            break
        
        source = cvss[b]
        source_audio = np.array([sample_audio(source)])
        source_text = np.array([source['text']])

        target_audio = np.array([librosa.resample(y = s['audio']['array'], orig_sr = 48000, target_sr = sampling_rate)])[0] # Common Voice's sr to model's sr
        target_text = np.array([s['sentence'].lower()])

        # Speech to text
        extracted_text = S2T(source_audio, sampling_rate)

        # Machine translation
        translated_text = MT(extracted_text)

        # Text to speech
        translated_audio = T2S(translated_text).detach().cpu().numpy()[0]

        # TODO:
        # Save audio translations
        save_wav(out_dir=args.out_dir, out_name=s['audio']['path'][6:], rate=sampling_rate, data=translated_audio)
        # Compute metrics of the batch
        bleu, charbleu, chrf, mcd = compute_metrics(
            target_text, target_audio, translated_audio, args.device
        )

        print("Batch", b)
        print("BLEU score :", bleu)
        print("charBLEU score :", charbleu)
        print("chrF score :", chrf)
        print("mcd score :", mcd)

        count += 1
        break


def main(args):
    # TODO:
    # Add more models!
    if args.stt_model == "fb-s2t-small":
        speech_to_text = GenericSTT(
            model="facebook/s2t-small-librispeech-asr", device=args.device
        )
    else:
        speech_to_text = None

    if args.mt_model == "m2m":
        machine_translation = M2M100(
            model="facebook/m2m100_418M",
            device=args.device,
            src_lang=args.src_lan,
            tgt_lan=args.tgt_lan,
        )
    elif args.mt_model == "hel":
        machine_translation = AutoMT(
            model="Helsinki-NLP/opus-mt-en-fr", device=args.device
        )
    else:
        machine_translation = None

    if args.tts_model == "fb-tts-fra":
        text_to_speech = VITS(model="facebook/mms-tts-fra", device=args.device)
    else:
        text_to_speech = None

    # Process in batches
    process_batches(
        speech_to_text,
        machine_translation,
        text_to_speech,
        16000,  # model's sampling rate
        args,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and predict")
    parser.add_argument(
        "--stt_model", type=str, help="name of the STT huggingface model"
    )
    parser.add_argument("--mt_model", type=str, help="name of the MT huggingface model")
    parser.add_argument(
        "--tts_model", type=str, help="name of the TTS huggingface model"
    )
    parser.add_argument("--src_lan", type=str, help="source language abbreviated")
    parser.add_argument("--tgt_lan", type=str, help="target language abbreviated")
    parser.add_argument(
        "--data_dir", type=str, metavar="DIR", help="directory containing the audios"
    )
    parser.add_argument(
        "--device",
        type=str,
        metavar="DIR",
        help="device to run the model on",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        metavar="DIR",
        help="path where the audios will be saved",
    )
    _args = parser.parse_args()
    main(_args)
