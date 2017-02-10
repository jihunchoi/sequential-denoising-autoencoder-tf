"""Encode sentences into fixed-size vectors using a trained model."""
import argparse

import tensorflow as tf
from tensorflow.contrib import layers, rnn

import sae
from config import ModelConfig
from util.data_generator import DataGenerator


def main():
    model_path = args.model
    config_path = args.config
    vocab_path = args.vocab
    test_data_path = args.test_data
    out_path = args.out
    batch_size = args.batch_size

    config = ModelConfig.load(config_path)

    data_generator = DataGenerator(
        data_path=test_data_path, vocab_path=vocab_path,
        eos_symbol='<EOS>', unk_symbol='<UNK>',
        omit_prob=0.0, swap_prob=0.0,
        batch_size=batch_size, max_length=10000, max_epoch=1)
    out_file = open(out_path, 'w')

    with tf.Graph().as_default():
        with tf.Session() as sess:
            inputs = tf.placeholder(dtype=tf.int32, shape=[None, None])
            inputs_length = tf.placeholder(dtype=tf.int32, shape=[None])

            vocab_size = len(data_generator.vocab)
            embeddings = tf.get_variable(name='embeddings',
                                         shape=[vocab_size, config.word_dim],
                                         dtype=tf.float32)

            with tf.variable_scope('decoder'):
                with tf.variable_scope('output') as output_scope:
                    # This variable-scope-trick is used to ensure that
                    # output_fn has a proper scope regardless of a caller's
                    # scope.
                    def output_fn(cell_outputs):
                        return layers.fully_connected(
                            inputs=cell_outputs, num_outputs=vocab_size,
                            activation_fn=None, scope=output_scope)

            rnn_cell = rnn.GRUBlockCell(config.sentence_dim)
            sent_vec = sae.encode(
                cell=rnn_cell, embeddings=embeddings,
                inputs=inputs, inputs_length=inputs_length, scope='encoder')

            saver = tf.train.Saver()
            saver.restore(sess=sess, save_path=model_path)

            for data_batch in data_generator:
                inputs_v, inputs_length_v, _, _ = data_batch
                sent_vec_v = sess.run(
                    fetches=sent_vec,
                    feed_dict={inputs: inputs_v,
                               inputs_length: inputs_length_v})
                for vec in sent_vec_v:
                    out_file.write(','.join('{:.5f}'.format(x) for x in vec))
                    out_file.write('\n')
    out_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Encode sentences into fixed-size vectors using '
                    'a trained model.')
    parser.add_argument('--model', required=True,
                        help='The path of a model file')
    parser.add_argument('--config', required=True,
                        help='The path of a configuration INI file')
    parser.add_argument('--vocab', required=True,
                        help='The path of a vocabulary file')
    parser.add_argument('--test-data', required=True,
                        help='The path of a test data file')
    parser.add_argument('--out', required=True,)
    parser.add_argument('--batch-size', default=32, type=int,
                        help='The size of a mini-batch')
    args = parser.parse_args()
    main()
