"""Preprocess the data.

It transforms all characters into lowercase letters and tokenize
each sentence into words using NLTK's word_tokenize method.
"""

import argparse
import logging

from nltk.tokenize import word_tokenize


LOG_FORMAT = '[%(levelname)-5s] %(asctime)-15s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=logging.INFO)


def read_and_preprocess_data(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, sentence in enumerate(f):
            words = word_tokenize(sentence.lower())
            data.append(words)
            if i > 0 and i % 10000 == 0:
                logging.info('Processed {} sentences'.format(i))
    return data


def write_data(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        for words in data:
            f.write(' '.join(words))
            f.write('\n')


def main():
    data_path = args.data
    out_path = args.out

    data = read_and_preprocess_data(data_path)
    write_data(data=data, path=out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess the data.')
    parser.add_argument('--data', required=True,
                        help='The path of a data file')
    parser.add_argument('--out', required=True,
                        help='The path to save the preprocessed file')
    args = parser.parse_args()
    main()
