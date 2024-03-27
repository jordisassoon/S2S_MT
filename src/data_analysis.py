import argparse
import pandas as pd
import numpy as np


def load_data():
    return pd.read_csv("data_analysis.py")


def sentence_length_dist(data):
    source = data["source"]
    target = data["target"]
    source_sentence_lengths = [len(x) for x in source]
    target_sentence_lengths = [len(x) for x in target]
    return source_sentence_lengths, target_sentence_lengths


def vocab_size(data):
    source = data["source"]
    target = data["target"]
    source_dict = set()
    target_dict = set()
    [[source_dict.add(y) for y in x.split(" ")] for x in source]
    [[target_dict.add(y) for y in x.split(" ")] for x in target]
    return source_dict, target_dict


def main(args):
    data = load_data()
    sentence_length_dist(data)
    vocab_size(data)

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
