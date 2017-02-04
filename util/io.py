"""Helper functions for IO."""


def read_data(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for sentence in f:
            words = sentence.split()
            data.append(words)
    return data


def read_vocab(vocab_path):
    vocab = dict()
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for line in f:
            w, i = line.split()
            vocab[w] = int(i)
    return vocab
