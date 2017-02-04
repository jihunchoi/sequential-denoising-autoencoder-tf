"""A data generator implementation."""
import random

import numpy as np

from .io import read_data, read_vocab


class DataGenerator(object):

    """A data generator class."""

    def pad_batch(self, data_batch, prepend_eos, append_eos):
        max_len = max(len(d) for d in data_batch)
        prefix = []
        suffix = []
        if prepend_eos:
            prefix = [self.word_to_id(self.eos_symbol)]
        if append_eos:
            suffix = [self.word_to_id(self.eos_symbol)]
        return [prefix + d + suffix + [0]*(max_len - len(d))
                for d in data_batch]

    def add_noise(self, word_ids):
        word_ids = word_ids.copy()
        # First, omit some words
        num_omissions = int(self.omit_prob * len(word_ids))
        inds_to_omit = np.random.permutation(len(word_ids))[:num_omissions]
        for i in inds_to_omit:
            word_ids[i] = self.word_to_id(self.unk_symbol)
        # Second, swap some of adjacent words
        num_swaps = int(self.swap_prob * (len(word_ids) - 1))
        inds_to_swap = np.random.permutation(len(word_ids) - 1)[:num_swaps]
        for i in inds_to_swap:
            word_ids[i], word_ids[i+1] = word_ids[i+1], word_ids[i]
        return word_ids

    def word_to_id(self, word):
        if word not in self.vocab:
            return self.vocab[self.unk_symbol]
        return self.vocab[word]

    def words_to_ids(self, words):
        return [self.word_to_id(word=w) for w in words]

    def id_to_word(self, id_):
        return self.reverse_vocab[id_]

    def ids_to_words(self, ids):
        return [self.id_to_word(id_=id_) for id_ in ids]

    def __init__(self, data_path, vocab_path, eos_symbol, unk_symbol,
                 omit_prob, swap_prob, batch_size, max_length, max_epoch):
        self.eos_symbol = eos_symbol
        self.unk_symbol = unk_symbol
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.omit_prob = omit_prob
        self.swap_prob = swap_prob

        self.data = [d for d in read_data(data_path)
                     if len(d) < max_length]
        self.vocab = read_vocab(vocab_path)
        self.reverse_vocab = dict((i, w) for w, i in self.vocab.items())

        self._epoch = 0
        self._progress_in_epoch = 0

    @property
    def progress(self):
        return self._epoch + self._progress_in_epoch

    def construct_data(self, words_batch):
        word_ids_batch = [self.words_to_ids(words=words)
                          for words in words_batch]
        length = np.array([len(d) for d in word_ids_batch],
                          dtype=np.int32)
        noise_word_ids_batch = [self.add_noise(word_ids)
                                for word_ids in word_ids_batch]
        inputs = np.array(
            self.pad_batch(noise_word_ids_batch,
                           prepend_eos=False, append_eos=True),
            dtype=np.int32)
        targets = np.array(
            self.pad_batch(word_ids_batch,
                           prepend_eos=True, append_eos=True),
            dtype=np.int32)
        inputs_length = length + 1
        targets_length = length + 2
        return inputs, inputs_length, targets, targets_length

    def __iter__(self):
        for self._epoch in range(self.max_epoch):
            random.shuffle(self.data)
            for i in range(0, len(self.data), self.batch_size):
                inputs, inputs_length, targets, targets_length = (
                    self.construct_data(self.data[i : i+self.batch_size]))
                yield inputs, inputs_length, targets, targets_length
                self._progress_in_epoch = i / len(self.data)

    def sample(self, num_samples):
        sample_inds = np.random.permutation(len(self.data))[:num_samples]
        words_sample = [self.data[i] for i in sample_inds]
        inputs, inputs_length, targets, targets_length = (
            self.construct_data(words_sample))
        return inputs, inputs_length, targets, targets_length
