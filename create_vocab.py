"""Create a vocabulary file from data."""
import argparse
from collections import Counter

from util.io import read_data


def create_vocab(data, max_size=None, word_count_path=None, vocab_path=None):
    word_counter = Counter()
    for words in data:
        word_counter.update(words)

    word_counts = word_counter.most_common(max_size)

    vocab = dict((w, i) for i, (w, _) in enumerate(word_counts))
    vocab['<UNK>'] = len(vocab)
    vocab['<EOS>'] = len(vocab)
    if word_count_path is not None:
        with open(word_count_path, 'w', encoding='utf-8') as f:
            for w, c in word_counts:
                f.write('{}\t{}\n'.format(w, c))
    if vocab_path is not None:
        with open(vocab_path, 'w', encoding='utf-8') as f:
            for w, i in vocab.items():
                f.write('{}\t{}\n'.format(w, i))
    return vocab


def main():
    data_path = args.data
    vocab_path = args.vocab
    word_count_path = args.word_count
    vocab_size = args.vocab_size

    data = read_data(data_path)
    create_vocab(data=data, max_size=vocab_size,
                 word_count_path=word_count_path, vocab_path=vocab_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create a vocabulary file from data.')
    parser.add_argument('--data', required=True,
                        help='The path of a data file')
    parser.add_argument('--vocab', required=True,
                        help='The path to save a vocabulary file')
    parser.add_argument('--word-count', required=True,
                        help='The path to save a word count file')
    parser.add_argument('--vocab-size', default=None, type=int,
                        help='The maximum size of a vocabulary set')
    args = parser.parse_args()
    main()
