import csv

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.model_selection import train_test_split
from typing import List, Tuple, Dict

from tensorflow.python.keras.callbacks import EarlyStopping


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


def build_model(preprocessor: str, bert: str) -> tf.keras.Model:
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
    net = outputs["pooled_output"]
    # net = tf.keras.layers.Dropout(0.05)(net)
    net = tf.keras.layers.Dense(1, activation="sigmoid", name="classifier")(net)
    return tf.keras.Model(text_input, net)


def train(model: tf.keras.Model, x_train: np.ndarray, y_train: np.ndarray):
    """
    Compiles and trains the given model.

    :param model: Model to be trained
    :param x_train: Training sentences
    :param y_train: Training labels
    """
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    classifier.fit(x_train, y_train)


def predict(model: tf.keras.Model, x_test: np.ndarray) -> List[int]:
    """
    Predicts binary labels.

    :param model: The model the predictions are based on
    :param x_test: The sentences to predict from
    :return: List of predicted labels
    """
    predictions = model.predict(x_test)
    return [1 if pr >= 0.5 else 0 for pr in predictions]


def score(gold_labels: List[int], pred_labels: List[int]) -> Dict[str, float]:
    """
    
    :param gold_labels: 
    :param pred_labels: 
    :return: 
    """
    return {'Accuracy': accuracy_score(gold_labels, pred_labels),
            'Precision': precision_score(gold_labels, pred_labels),
            'Recall': recall_score(gold_labels, pred_labels),
            'F1': f1_score(gold_labels, pred_labels, average='binary'),
            'F1 macro': f1_score(gold_labels, pred_labels, average='macro')}


if __name__ == "__main__":
    # ENGLISH
    en_X_train, en_X_test, en_y_train, en_y_test = load_subtask1_data("en")
    classifier = build_model("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
                             "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3")
    train(classifier, np.array(en_X_train), np.array(en_y_train))
    predicted_labels = predict(classifier, np.array(en_X_test))
    print(score(en_y_test, predicted_labels))

    # Load French language data
    fr_X_train, fr_X_test, fr_y_train, fr_y_test = load_subtask1_data("fr")

    # Load Italian language data
    it_X_train, it_X_test, it_y_train, it_y_test = load_subtask1_data("it")
