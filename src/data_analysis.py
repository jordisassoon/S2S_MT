import argparse
import pandas as pd
import numpy as np
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import gensim
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.tokenize import word_tokenize
import pyLDAvis.gensim
import re


def load_data():
    return pd.read_csv("source_target_texts.csv")


def remove_number_words(text):
    # Define a list of number words
    number_words = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
                    'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen',
                    'eighteen', 'nineteen', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy',
                    'eighty', 'ninety', 'hundred', 'thousand', 'million', 'billion', 'trillion']

    # Create a regular expression pattern to match number words
    pattern = r'\b(?:%s)\b' % '|'.join(number_words)

    # Use regex to remove number words from the text
    return re.sub(pattern, '', text)


def sentence_length_dist(data):
    source = [remove_number_words(str(x)) for _, x in data["source"].items()]
    target = [remove_number_words(str(x)) for _, x in data["target"].items()]
    source_sentence_lengths = [len(x.split(" ")) for x in source]
    target_sentence_lengths = [len(x.split(" ")) for x in target]

    counts, bins = np.histogram(source_sentence_lengths, bins=20)
    plt.hist(bins[:-1], bins, weights=counts)
    plt.show()

    counts, bins = np.histogram(target_sentence_lengths, bins=20)
    plt.hist(bins[:-1], bins, weights=counts)
    plt.show()


def vocab_size(data):
    source = [remove_number_words(str(x)) for _, x in data["source"].items()]
    target = [remove_number_words(str(x)) for _, x in data["target"].items()]
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


def topic_modeling(data):
    stop_words = set(stopwords.words('english'))
    corpus = []
    lem = WordNetLemmatizer()
    source = [remove_number_words(str(x)) for _, x in data["source"].items()]
    for sentence in source:
        words = [w for w in word_tokenize(str(sentence)) if (w not in stop_words)]
        words = [lem.lemmatize(w) for w in words if len(w) > 2]
        corpus.append(words)

    dic = gensim.corpora.Dictionary(corpus)
    bow_corpus = [dic.doc2bow(doc) for doc in corpus]
    lda_model = gensim.models.LdaMulticore(bow_corpus,
                                           num_topics=4,
                                           id2word=dic,
                                           passes=10,
                                           workers=2)
    lda_model.show_topics()
    vis = pyLDAvis.gensim.prepare(lda_model, bow_corpus, dic)
    pyLDAvis.show(vis)


def main(args):
    data = load_data()
    sentence_length_dist(data)
    print("Number of tokens in source and target:", vocab_size(data))
    word_cloud(data)
    topic_modeling(data)


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
