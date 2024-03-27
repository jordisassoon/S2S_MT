import argparse
from datasets import load_dataset

from models.STT.GenericSTT import GenericSTT
from models.MT.M2M100 import M2M100
from models.MT.AutoMT import AutoMT
from models.TTS.VITS import VITS

from utils.metrics import compute_metrics, compute_all_text_metrics
from utils.save import save_batch

from tqdm import tqdm
import librosa
import pandas as pd

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
        "mozilla-foundation/common_voice_4_0", "fr", split="validation", trust_remote_code=True, token=token, cache_dir='/data/kchardon-22/datasets'
    )
    max_size = args.max_size
    if max_size is None or max_size > len(cvss) :
        max_size = len(cvss)

    if max_size < batch_size:
        batch_size = max_size

    # Scores over all the samples
    bleu_all = 0
    charbleu_all = 0
    chrf_all = 0
    mcd_all = 0

    columns = ['file_name','source_text','extracted_text', 'S2T_BLEU', 'S2T_CHARBLEU', 'target_text', 'translated_text', 'MT_BLEU', 'MT_CHARBLEU']
    steps_scores = pd.DataFrame(columns=columns)

    columns = ['file_name', 'target_text', 'transcribed_target', 'bleu', 'charbleu', 'chrf', 'mcd']
    s2s_scores = pd.DataFrame(columns=columns)

    for b, i in tqdm(enumerate(range(0, max_size, batch_size))):
        cvss_batch = [cvss[index] for index in range(i, i + batch_size)]
        cv_batch = [common_voice[index] for index in range(i, i + batch_size)]
        
        # Source, English
        source_audio = [sample_audio(source) for source in cvss_batch]
        source_text = [source['text'] for source in cvss_batch]

        # Target, French
        target_audio = [sample_audio(target) for target in cv_batch]
        target_text = [target['sentence'].lower() for target in cv_batch]

        # Speech to text
        extracted_text = S2T(source_audio, sampling_rate)
        # Metrics on text between source_text and extracted_text
        # bleu, charbleu, _ = compute_all_text_metrics(source_text, extracted_text)

        # Machine translation
        translated_text = MT(extracted_text)
        # Metrics on text between translated_text and target_text
        # bleu, charbleu, _ = compute_all_text_metrics(target_text, translated_text)

        # Text to speech
        translated_audio = T2S(translated_text).detach().cpu().numpy()

        # Save audio translations
        save_batch(out_dir=args.out_dir, out_name=[source['id'] for source in cvss_batch], rate=sampling_rate, data=translated_audio)

        # Compute all the metrics for translated_audio
        bleu, charbleu, chrf, mcd = compute_metrics(
            target_text, target_audio, translated_audio, args.device
        )

        '''
        print("Batch", b)
        print("BLEU score :", bleu)
        print("charBLEU score :", charbleu)
        print("chrF score :", chrf)
        print("mcd score :", mcd)
        '''

        bleu_all += bleu
        charbleu_all += charbleu
        chrf_all += chrf
        mcd_all += mcd

        count += batch_size
        break
    
    bleu_all /= count
    charbleu_all /= count
    chrf_all /= count
    mcd_all /= count

    print("BLEU score :", bleu)
    print("charBLEU score :", charbleu)
    print("chrF score :", chrf)
    print("mcd score :", mcd)


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
    parser.add_argument(
        "--max_size",
        type=int,
        help="number of samples to process",
    )
    _args = parser.parse_args()
    main(_args)
