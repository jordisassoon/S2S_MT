import argparse
import pandas as pd
import numpy as np
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt


def load_data():
    # return pd.read_csv("data_analysis.py")
    return pd.DataFrame(data={'source': ["hello"], 'target': ["bonjour"]})

def sentence_length_dist(data):
    source = data["source"]
    target = data["target"]
    source_sentence_lengths = [len(x.split(" ")) for x in source]
    target_sentence_lengths = [len(x.split(" ")) for x in target]

    counts, bins = np.histogram(source_sentence_lengths)
    plt.stairs(counts, bins)
    plt.show()

    counts, bins = np.histogram(target_sentence_lengths)
    plt.stairs(counts, bins)
    plt.show()


def vocab_size(data):
    source = data["source"]
    target = data["target"]
    source_dict = set()
    target_dict = set()
    [[source_dict.add(y) for y in x.split(" ")] for x in source]
    [[target_dict.add(y) for y in x.split(" ")] for x in target]
    return len(source_dict), len(target_dict)


def word_cloud(data):
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=50,
        max_font_size=30,
        scale=3,
        random_state=1)

    wordcloud = wordcloud.generate(str(data.source))

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')

    plt.imshow(wordcloud)
    plt.show()


def main(args):
    data = load_data()
    sentence_length_dist(data)
    print("Number of tokens in source and target:", vocab_size(data))
    word_cloud(data)


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
