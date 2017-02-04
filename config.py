"""Model configuration object."""
from configparser import ConfigParser


class ModelConfig(object):

    def __init__(self, word_dim, sentence_dim, omit_prob, swap_prob):
        self.word_dim = word_dim
        self.sentence_dim = sentence_dim
        self.omit_prob = omit_prob
        self.swap_prob = swap_prob

    def save(self, path):
        config = ConfigParser()
        config['model'] = {'word_dim': self.word_dim,
                           'sentence_dim': self.sentence_dim,
                           'omit_prob': self.omit_prob,
                           'swap_prob': self.swap_prob}
        with open(path, 'w') as f:
            config.write(f)

    @classmethod
    def load(cls, path):
        config = ConfigParser()
        config.read(path)
        word_dim = config['model'].getint('word_dim')
        sentence_dim = config['model'].getint('sentence_dim')
        omit_prob = config['model'].getfloat('omit_prob')
        swap_prob = config['model'].getfloat('swap_prob')
        return cls(word_dim=word_dim, sentence_dim=sentence_dim,
                   omit_prob=omit_prob, swap_prob=swap_prob)
