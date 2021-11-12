from sklearn.model_selection import train_test_split
import csv
import numpy as np
from collections import Counter
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
import random
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


def word_counter(corpus):
    """Function that creates a counter object on the given corpus that stores
        the number of occurrences of each word in that corpus as key-value pairs
        (word - num_of_occurrences)
        Parameters:
        -----------
        corpus:	list of texts
        Return value:	word counter object
    """
    count = Counter()
    for text in corpus:
        for word in text.split():
            count[word] += 1

    return count


def read_data(filename, sentence, label):
    """
   Read the data file and extract the values from sentence and labels
   :param filename: Name/path of tsv file containing columns for sentence and labels
   :param sentence: list where sentences from file will be added
   :param label: list where labels from file will be added
   """
    with open(filename, 'rt', encoding='utf-8') as f:
        csvr = csv.DictReader(f, delimiter='\t')
        for row in csvr:
            sentence.append(row['Sentence'])
            label.append(int(row['Labels']))


def load_data(language):
    """
    Load data provided in train files for subtask 1 picked from a random fold and return train and test splits.
    :param language: language name picked for the model
    :return: tuple (train_sent, test_sent, train_label, test_label) with shuffled data
    """
    sentence, label = [], []
    num = random.randint(0, 2)  # pick random fold
    read_data(f"data/train/train_subtask-1/{language}/{language.capitalize()}-Subtask1-fold_{num}.tsv", sentence, label)
    label = np.array(label)
    train_sent, test_sent, train_label, test_label = train_test_split(sentence, label, random_state=0, test_size=0.2)
    return train_sent, test_sent, train_label, test_label


def binary_prediction(trn_sent, trn_label, tst_sent, tst_label):
    """
    Build a binary classifier using a bidirectional LSTM
    :param trn_sent: training sentences as split in load_data()
    :param trn_label: training labels as split in load_data()
    :param tst_sent: test sentences as split in load_data()
    :param tst_label: test labels as split in load_data()
    :return: Binary predictions
    """
    counter = word_counter(trn_sent)
    num_words = len(counter)
    max_len = 12  # max 12 words

    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(trn_sent)

    # prepare train data
    train_seqs = tokenizer.texts_to_sequences(trn_sent)
    train_padded = pad_sequences(train_seqs, maxlen=max_len, padding='post', truncating='post')

    # prepare test data (data for which we want to make predictions later)
    test_seqs = tokenizer.texts_to_sequences(tst_sent)
    test_padded = pad_sequences(test_seqs, maxlen=max_len, padding='post', truncating='post')
    model = Sequential()
    model.add(Embedding(num_words, 32, input_length=max_len))
    model.add(Bidirectional(LSTM(64, dropout=0.1, return_sequences=True)))
    model.add(Bidirectional(LSTM(32, dropout=0.1)))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # train the model
    model.fit(train_padded, trn_label, epochs=35)

    # predict on the test set and return a list of binary predictions
    predictions = model.predict(test_padded)
    predicted_labels = [1 if pr >= 0.5 else 0 for pr in predictions]
    accuracy = accuracy_score(tst_label, predicted_labels)
    precision = precision_score(tst_label, predicted_labels)
    recall = recall_score(tst_label, predicted_labels)
    f1 = f1_score(tst_label, predicted_labels,average='binary')
    f1_macro = f1_score(tst_label, predicted_labels, average='macro')

    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'F1 macro': f1_macro}


if __name__ == "__main__":
    # ENGLISH
    en_X_train, en_X_test, en_y_train, en_y_test = load_data("en")

    # Load French language data
    fr_X_train, fr_X_test, fr_y_train, fr_y_test = load_data("fr")

    # Load Italian language data
    it_X_train, it_X_test, it_y_train, it_y_test = load_data("it")

    print("English", binary_prediction(en_X_train, en_y_train, en_X_test, en_y_test))
    print("French", binary_prediction(fr_X_train, fr_y_train, fr_X_test, fr_y_test))
    print("Italian", binary_prediction(it_X_train, it_y_train, it_X_test, it_y_test))
