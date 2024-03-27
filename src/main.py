import argparse
from datasets import load_dataset

from models.STT.GenericSTT import GenericSTT
from models.STT.WhisperSTT import WhisperSTT
from models.MT.M2M100 import M2M100
from models.MT.AutoMT import AutoMT
from models.TTS.VITS import VITS

from tools.metrics import compute_metrics, compute_all_text_metrics
from tools.save import save_batch

from tqdm import tqdm
import librosa
import pandas as pd
import numpy as np

from token_hf import token


def sample_audio(audio):
    speech_array, _ = librosa.load(audio["audio"]["path"], sr=16000) # model's sampling rate
    return speech_array


def process_batches(S2T, MT, T2S, sampling_rate, args, batch_size=5):
    # Load datasets
    # Source, English
    cvss = load_dataset(
        "google/cvss", "cvss_c", languages=["fr"], split="validation", trust_remote_code=True
    )
    
    # Target, French
    common_voice = load_dataset(
        "mozilla-foundation/common_voice_4_0", "fr", split="validation", trust_remote_code=True, token=token, cache_dir='/nvme/kchardon-22/datasets'
    )
    max_size = args.max_size
    if max_size is None or max_size > len(cvss):
        max_size = len(cvss)

    if max_size < batch_size:
        batch_size = max_size

    # Get all the scores and outputs
    file_name_all = []
    source_text_all = []
    extracted_text_all = []
    S2T_bleu = []
    S2T_charbleu= []
    target_text_all = []
    translated_text_all = []
    MT_bleu = []
    MT_charbleu = []

    transcribed_target_all = []
    bleu_all = []
    charbleu_all = []
    chrf_all = []
    mcd_all = []

    for b, i in tqdm(enumerate(range(0, max_size, batch_size))):
        print("Batch", b)
        cvss_batch = [cvss[index] for index in range(i, i + batch_size)]
        cv_batch = [common_voice[index] for index in range(i, i + batch_size)]
        
        # Source, English
        source_audio = [sample_audio(source) for source in cvss_batch]
        source_text = [source['text'] for source in cvss_batch]

        # Target, French
        target_audio = [sample_audio(target) for target in cv_batch]
        target_text = [target['sentence'].lower() for target in cv_batch]

        # Save
        file_name_all.extend([source['id'] for source in cvss_batch])
        source_text_all.extend(source_text)
        target_text_all.extend(target_text)

        # Speech to text
        extracted_text = S2T(source_audio, sampling_rate)
        # Metrics on text between source_text and extracted_text
        bleu, charbleu, _ = compute_all_text_metrics(source_text, extracted_text)
        extracted_text_all.extend(extracted_text)
        S2T_bleu.append(bleu)
        S2T_charbleu.append(charbleu)
        print('Extraction bleu :', bleu)
        print('Extraction charbleu :', charbleu)

        # Machine translation
        translated_text = MT(extracted_text)
        # Metrics on text between translated_text and target_text
        bleu, charbleu, _ = compute_all_text_metrics(target_text, translated_text)
        translated_text_all.extend(translated_text)
        MT_bleu.append(bleu)
        MT_charbleu.append(charbleu)
        print('Translation bleu :', bleu)
        print('Translation charbleu :', charbleu)

        # Text to speech
        translated_audio = [audio.detach().cpu().numpy() for audio in T2S(translated_text)]

        # Save audio translations
        save_batch(out_dir=args.out_dir, out_name=[source['id'] for source in cvss_batch], rate=sampling_rate, batch=translated_audio)

        # Compute all the metrics for translated_audio
        bleu, charbleu, chrf, mcd, transcribed_target = compute_metrics(
            target_text, target_audio, translated_audio, args.device
        )

        print("BLEU score :", bleu)
        print("charBLEU score :", charbleu)
        print("chrF score :", chrf)
        print("mcd score :", mcd)

        transcribed_target_all.extend(transcribed_target)
        bleu_all.append(bleu)
        charbleu_all.append(charbleu)
        chrf_all.append(chrf)
        mcd_all.append(mcd)


        break
    
    print("TOTAL :")
    print("BLEU score :", np.mean(bleu_all * batch_size))
    print("charBLEU score :", np.mean(charbleu_all * batch_size))
    print("chrF score :", np.mean(chrf_all * batch_size))
    print("mcd score :", np.mean(mcd_all * batch_size))

    print('Extraction bleu :', np.mean(S2T_bleu * batch_size))
    print('Extraction charbleu :', np.mean(charbleu * batch_size))

    print('Translation bleu :', np.mean(MT_bleu * batch_size))
    print('Translation charbleu :', np.mean(MT_charbleu * batch_size))

    columns = {'file_name' : file_name_all,'source_text' : source_text_all,'extracted_text' : extracted_text_all, 'target_text' : target_text_all, 'translated_text' : translated_text_all}
    steps_scores = pd.DataFrame(columns)
    steps_scores.to_csv(args.out_dir+'/'+'steps_outputs.csv')

    columns = {'file_name' : file_name_all,'source_text' : source_text_all, 'target_text' : target_text_all, 'transcribed_target' : transcribed_target_all}
    s2s_scores = pd.DataFrame(columns)
    s2s_scores.to_csv(args.out_dir+'/'+'s2s_outputs.csv')



def main(args):
    # TODO:
    # Add more models!
    if args.stt_model == "fb-s2t-small":
        speech_to_text = GenericSTT(
            model="facebook/s2t-small-librispeech-asr", device=args.device
        )
    elif args.stt_model == "whisper-small":
        speech_to_text = WhisperSTT(
            model="openai/whisper-small", device=args.device
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
    elif args.mt_model == "hel" and args.tgt_lan == "fr":
        machine_translation = AutoMT(
            model="Helsinki-NLP/opus-mt-en-fr", device=args.device
        )
    elif args.mt_model == "hel" and args.tgt_lan == "lv":
        machine_translation = AutoMT(
            model="Helsinki-NLP/opus-mt-tc-big-en-lv", device=args.device
        )
    else:
        machine_translation = None

    if args.tts_model == "fb-tts" and args.tgt_lan == "fr":
        text_to_speech = VITS(model="facebook/mms-tts-fra", device=args.device)
    elif args.tts_model == "fb-tts" and args.tgt_lan == "lv":
        text_to_speech = VITS(model="facebook/mms-tts-lav", device=args.device)
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
    parser.add_argument(
        "--max_size",
        type=int,
        help="number of samples to process",
    )
    _args = parser.parse_args()
    main(_args)
