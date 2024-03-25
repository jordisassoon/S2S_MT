import argparse
from datasets import load_dataset

from models.STT.GenericSTT import GenericSTT
from models.MT.M2M100 import M2M100
from models.MT.AutoMT import AutoMT
from models.TTS.VITS import VITS

from utils.metrics import compute_metrics
from utils.save import save_batch


def main(args):
    # TODO:
    # Replace this with a custom loaded dataset
    dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    inputs = dataset[0]["audio"]["array"]
    sampling_rate = dataset[0]["audio"]["sampling_rate"]

    # TODO:
    # Add more models!
    if args.stt_model == "fb-s2t-small":
        speech_to_text = GenericSTT(model="facebook/s2t-small-librispeech-asr", device=args.device)
    else:
        speech_to_text = None

    if args.mt_model == "m2m":
        machine_translation = M2M100(model="facebook/m2m100_418M", device=args.device, tgt_lan=args.tgt_lan)
    elif args.mt_model == "hel":
        machine_translation = AutoMT(model="Helsinki-NLP/opus-mt-en-fr", device=args.device)
    else:
        machine_translation = None

    if args.tts_model == "fb-tts-fra":
        text_to_speech = VITS(model="facebook/mms-tts-fra", device=args.device)
    else:
        text_to_speech = None

    # TODO:
    # Process in batches
    extracted_text = speech_to_text(inputs, sampling_rate)
    print(extracted_text)

    translated_text = machine_translation(extracted_text)
    print(translated_text)

    translated_audio = text_to_speech(translated_text)[0].detach().numpy()

    # TODO:
    # Make a saving tool
    save_batch(out_dir=args.out_dir, rate=sampling_rate, batch=[translated_audio])

    # Test of metrics
    outputs = [["Monsieur Kilter est l'apôtre des classes moyennes et nous sommes heureux d'accueillir son évangile"]]
    bleu, charbleu, chrf, mcd = compute_metrics(outputs, translated_audio, "./real_out.wav", "./out.wav", args.device)

    print("BLEU score :", bleu)
    print("charBLEU score :", charbleu)
    print("chrF score :", chrf)
    print("mcd score :", mcd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and predict")
    parser.add_argument(
        "--stt_model", type=str, help="name of the STT huggingface model"
    )
    parser.add_argument(
        "--mt_model", type=str, help="name of the MT huggingface model"
    )
    parser.add_argument(
        "--tts_model", type=str, help="name of the TTS huggingface model"
    )
    parser.add_argument(
        "--src_lan", type=str, help="source language abbreviated"
    )
    parser.add_argument(
        "--tgt_lan", type=str, help="target language abbreviated"
    )
    parser.add_argument(
        "--data_dir", type=str, metavar="DIR", help="directory containing the data"
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
