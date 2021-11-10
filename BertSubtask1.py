import bert
import csv

from bert import run_classifier
from sklearn.model_selection import train_test_split


def read_binary_label_data(filename, sentences, labels):
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


def load_subtask1_data(language):
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


def create_input_examples(x, y):
    """
    Creates BERT InputExamples from list data.

    :param x: A list of sentences
    :param y: A list of labels corresponding to the sentences
    :return:
    """
    input_examples = []
    for i in range(len(x)):
        input_examples.append(bert.run_classifier.InputExample(guid=None, text_a=x[i], text_b=None, label=y[i]))
    return input_examples


if __name__ == "__main__":
    # Load English language data
    en_X_train, en_X_test, en_y_train, en_y_test = load_subtask1_data("en")
    en_train_InputExamples = create_input_examples(en_X_train, en_y_train)
    en_test_InputExamples = create_input_examples(en_X_test, en_y_test)

    # Load French language data
    fr_X_train, fr_X_test, fr_y_train, fr_y_test = load_subtask1_data("fr")

    # Load Italian language data
    it_X_train, it_X_test, it_y_train, it_y_test = load_subtask1_data("it")
