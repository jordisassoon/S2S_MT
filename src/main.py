import argparse
import torch
import scipy
from datasets import load_dataset

from models.STT import SpeechToText
from models.MT import MachineTranslation
from models.TTS import TextToSpeech

from metrics import compute_metrics


def main(args):
    dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

    inputs = dataset[0]["audio"]["array"]
    sampling_rate = dataset[0]["audio"]["sampling_rate"]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    speech_to_text = SpeechToText(model="facebook/s2t-small-librispeech-asr", device=device)
    machine_translation = MachineTranslation(model="facebook/m2m100_418M", device=device, src_lang="en", tgt_lan="fr")
    text_to_speech = TextToSpeech(model="facebook/mms-tts-fra", device=device)

    extracted_text = speech_to_text(inputs, sampling_rate)
    print(extracted_text)
    
    translated_text = machine_translation(extracted_text)
    print(translated_text)

    translated_audio = text_to_speech(translated_text)[0].detach().numpy()
    scipy.io.wavfile.write("out.wav", rate=sampling_rate, data=translated_audio)

    # Test of metrics
    outputs = [["Monsieur Kilter est l'apôtre des classes moyennes et nous sommes heureux d'accueillir son évangile"]]
    bleu, charbleu, chrf, mcd = compute_metrics(outputs, translated_audio, "real_out.wav", "out.wav", device)

    print("BLEU score :", bleu)
    print("charBLEU score :", charbleu)
    print("chrF score :", chrf)
    print("mcd score :", mcd)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and predict")
    parser.add_argument(
        "--model", type=str, help="name of the underlying model"
    )
    parser.add_argument(
        "--model_config", type=str, metavar="DIR", help="path to yaml"
    )
    parser.add_argument(
        "--train_data", type=str, metavar="DIR", help="npy filename"
    )
    parser.add_argument(
        "--labels", type=str, metavar="DIR", help="csv filename"
    )
    parser.add_argument(
        "--label_column", type=str, metavar="DIR", help="label column"
    )
    parser.add_argument(
        "--test_data", type=str, metavar="DIR", help="npy filename"
    )
    parser.add_argument(
        "--device",
        type=str,
        metavar="DIR",
        help="GPU or CPU to run the model on",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        metavar="DIR",
        help="path where the predictions will be saved",
    )
    _args = parser.parse_args()
    main(_args)