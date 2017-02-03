"""Helper functions for IO."""


def read_data(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for sentence in f:
            words = sentence.split()
            data.append(words)
    return data
