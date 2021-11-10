import csv
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

from sklearn.model_selection import train_test_split
from typing import List, Tuple


def read_binary_label_data(filename: str, sentences: List[str], labels: List[int]):
    """
    Adds the data from the given filename to the lists of sentences and labels.

    :param filename: Path to a tsv file containing a list of sentences and corresponding labels
    :param sentences: List of sentences to append the new data to.
    :param labels: List of labels to append the new data to.
    """
    with open(filename, encoding="UTF-8") as file:
        read_tsv = csv.reader(file, delimiter="\t")
        next(read_tsv, None)  # skip header
        for row in read_tsv:
            sentences.append(row[1])
            labels.append(int(row[2]))


def load_subtask1_data(language: str) -> Tuple[List[str], List[int], List[str], List[int]]:
    """
    Loads the data from the subtask1 train file and returns shuffled train and test split.

    :param language: Language to load the data for
    :return: A tuple (x_train, x_test, y_train, y_test) that contains the data shuffled and split in train and test
    """
    sentences = []
    labels = []
    for i in range(3):
        read_binary_label_data(f"data/train/train_subtask-1/{language}/{language.capitalize()}-Subtask1-fold_{i}.tsv",
                               sentences, labels)
    x_train, x_test = train_test_split(sentences, random_state=0)
    y_train, y_test = train_test_split(labels, random_state=0)
    return x_train, x_test, y_train, y_test


def build_model(preprocessor: str, bert: str):
    """
    Builds a binary classifier based on the pretrained models.

    :param preprocessor: Path to the pretrained BERT preprocessor
    :param bert: Path to the pretrained BERT language model
    :return: Binary classifier
    """
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer(preprocessor, name="preprocessor")
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(bert, name="BERT")
    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
    return tf.keras.Model(text_input, net)


if __name__ == "__main__":
    # ENGLISH
    en_X_train, en_X_test, en_y_train, en_y_test = load_subtask1_data("en")
    classifier = build_model("https://tfhub.dev/tensorflow/bert_en_cased_preprocess/3",
                             "https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/3")
    classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    # classifier.fit()

    # Load French language data
    fr_X_train, fr_X_test, fr_y_train, fr_y_test = load_subtask1_data("fr")

    # Load Italian language data
    it_X_train, it_X_test, it_y_train, it_y_test = load_subtask1_data("it")
