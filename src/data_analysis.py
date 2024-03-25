import argparse

""" TODO:
Step 1: Basic info (audio length, audio variance, sentence length, vocab size, etc)
Step 2: Recognizing useless info (noise, jargon)
Step 3: Preprocess the audios if possible, or add preprocessing steps in the pipeline before translation
"""


def audio_length_dist():
    return NotImplementedError


def sentence_length_dist():
    return NotImplementedError


def audio_variance():
    return NotImplementedError


def eval_noise():
    return NotImplementedError


def main(args):
    audio_length_dist()
    sentence_length_dist()
    audio_variance()
    eval_noise()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and predict")
    parser.add_argument(
        "--data_dir", type=str, help="relative directory of the audios samples"
    )
    parser.add_argument("--audios_dir", type=str, help="folder containing the audios")
    parser.add_argument(
        "--transcriptions_dir", type=str, help="folder containing the transcriptions"
    )
    _args = parser.parse_args()
    main(_args)
