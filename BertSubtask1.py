import csv
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


if __name__ == "__main__":

    # Load English language data
    en_sentences = []
    en_labels = []
    for i in range(3):
        read_binary_label_data(f"data/train/train_subtask-1/en/En-Subtask1-fold_{i}.tsv", en_sentences, en_labels)
    en_X_train, en_X_test = train_test_split(en_sentences, random_state=0)
    en_y_train, en_y_test = train_test_split(en_labels, random_state=0)

    # Load French language data
    fr_sentences = []
    fr_labels = []
    for i in range(3):
        read_binary_label_data(f"data/train/train_subtask-1/fr/Fr-Subtask1-fold_{i}.tsv", fr_sentences, fr_labels)
    fr_X_train, fr_X_test = train_test_split(fr_sentences, random_state=1)
    fr_y_train, fr_y_test = train_test_split(fr_labels, random_state=1)

    # Load Italian language data
    it_sentences = []
    it_labels = []
    for i in range(3):
        read_binary_label_data(f"data/train/train_subtask-1/it/It-Subtask1-fold_{i}.tsv", it_sentences, it_labels)
    it_X_train, it_X_test = train_test_split(it_sentences, random_state=2)
    it_y_train, it_y_test = train_test_split(it_labels, random_state=2)
