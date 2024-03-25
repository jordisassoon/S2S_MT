import argparse
from datasets import load_dataset

from models.STT.GenericSTT import GenericSTT
from models.MT.M2M100 import M2M100
from models.MT.AutoMT import AutoMT
from models.TTS.VITS import VITS

from utils.metrics import compute_metrics
from utils.save import save_batch

from tqdm import tqdm
import librosa


def sample_audio(audio):
    speech_array, sampling_rate = librosa.load(audio["file"], sr=16_000)
    return speech_array


def process_batches(dataset, sampling_rate, S2T, MT, T2S, args, batch_size=2):
    if len(dataset) < batch_size:
        batch_size = len(dataset)

    # TODO:
    # Scores over all the batches

    for b, i in tqdm(enumerate(range(0, len(dataset), batch_size))):
        batch = [dataset[index] for index in range(i, i + batch_size)]

        # Batch of inputs
        inputs = [sample_audio(sample) for sample in batch]

        # Speech to text
        extracted_text = S2T(inputs, sampling_rate)

        # Machine translation
        translated_text = MT(extracted_text)

        # Text to speech
        translated_audio = [audio.detach().numpy() for audio in T2S(translated_text)]

        # TODO:
        # Save audio translations
        save_batch(out_dir=args.out_dir, rate=sampling_rate, batch=translated_audio)

        # Compute metrics of the batch
        outputs = [sample["text"] for sample in batch]
        bleu, charbleu, chrf, mcd = compute_metrics(
            outputs, translated_audio, "./real_out.wav", "./out.wav", args.device
        )

        print("Batch", b)
        print("BLEU score :", bleu)
        print("charBLEU score :", charbleu)
        print("chrF score :", chrf)
        print("mcd score :", mcd)


def main(args):
    dataset = load_dataset(
        "google/cvss", "cvss_c", languages=["fr"], split="validation"
    )

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
        dataset,
        16000,  # model's sampling rate
        speech_to_text,
        machine_translation,
        text_to_speech,
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
