"""Train the SDAE model."""
import argparse
import os
from datetime import datetime

import tensorflow as tf
from tensorflow.contrib import layers, rnn, slim
from tensorflow.contrib.framework import get_or_create_global_step

import sae
from config import ModelConfig
from util.data_generator import DataGenerator


logging = tf.logging
logging.set_verbosity(logging.INFO)


def main():
    data_path = args.data
    vocab_path = args.vocab
    save_dir = args.save_dir
    word_dim = args.word_dim
    sentence_dim = args.sentence_dim
    omit_prob = args.omit_prob
    swap_prob = args.swap_prob
    config_path = args.config
    batch_size = args.batch_size
    max_epoch = args.max_epoch
    max_length = args.max_length

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Check whether all needed options are given
    if config_path is not None:
        assert (word_dim is None and sentence_dim is None
                and omit_prob is None and swap_prob is None), (
            'Model hyperparameter options must not be provided when '
            'the "config" option is given.')
        config = ModelConfig.load(config_path)
    else:
        assert not (word_dim is None or sentence_dim is None
                    or omit_prob is None or swap_prob is None), (
            'All model hyperparameter options must be provided when '
            'the "config" option is not given.')
        config = ModelConfig(word_dim=word_dim, sentence_dim=sentence_dim,
                             omit_prob=omit_prob, swap_prob=swap_prob)
        config_path = os.path.join(save_dir, 'config.ini')
        config.save(config_path)

    logging.info('Initializing the data generator...')
    data_generator = DataGenerator(
        data_path=data_path, vocab_path=vocab_path,
        eos_symbol='<EOS>', unk_symbol='<UNK>',
        omit_prob=config.omit_prob, swap_prob=config.swap_prob,
        batch_size=batch_size, max_length=max_length, max_epoch=max_epoch)
    with tf.Graph().as_default() as graph:
        with tf.Session() as sess:
            logging.info('Building the model...')
            # Placeholders
            inputs = tf.placeholder(dtype=tf.int32, shape=[None, None],
                                    name='inputs')
            inputs_length = tf.placeholder(dtype=tf.int32, shape=[None],
                                           name='inputs_length')
            targets = tf.placeholder(dtype=tf.int32, shape=[None, None],
                                     name='targets')
            targets_length = tf.placeholder(dtype=tf.int32, shape=[None],
                                            name='targets_length')

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
            encoder_state = sae.encode(
                cell=rnn_cell, embeddings=embeddings,
                inputs=inputs, inputs_length=inputs_length, scope='encoder')
            decoder_outputs = sae.decode_train(
                cell=rnn_cell, embeddings=embeddings,
                encoder_state=encoder_state, targets=targets,
                targets_length=targets_length, scope='decoder')
            generated = sae.decode_inference(
                cell=rnn_cell, embeddings=embeddings,
                encoder_state=encoder_state, output_fn=output_fn,
                vocab_size=vocab_size,
                bos_id=data_generator.vocab['<EOS>'],
                eos_id=data_generator.vocab['<EOS>'],
                max_length=max_length,
                scope='decoder', reuse=True)
            loss = sae.loss(decoder_outputs=decoder_outputs,
                            output_fn=output_fn,
                            targets=targets, targets_length=targets_length)

            global_step = get_or_create_global_step()
            train_op = slim.optimize_loss(
                loss=loss, global_step=global_step, learning_rate=None,
                optimizer=tf.train.AdamOptimizer(), clip_gradients=5.0)

            summary_writer = tf.summary.FileWriter(
                logdir=os.path.join(save_dir, 'log'), graph=graph)
            summary = tf.summary.merge_all()

            tf.get_variable_scope().set_initializer(
                tf.random_normal_initializer(mean=0.0, stddev=0.01))
            tf.global_variables_initializer().run()

            saver = tf.train.Saver(max_to_keep=20)

            logging.info('Training starts!')
            for data_batch in data_generator:
                (inputs_v, inputs_length_v,
                 targets_v, targets_length_v) = data_batch
                summary_v, global_step_v, _ = sess.run(
                    fetches=[summary, global_step, train_op],
                    feed_dict={inputs: inputs_v,
                               inputs_length: inputs_length_v,
                               targets: targets_v,
                               targets_length: targets_length_v})
                summary_writer.add_summary(summary=summary_v,
                                           global_step=global_step_v)
                if global_step_v % 100 == 0:
                    logging.info('{} Iter #{}, Epoch {:.2f}'
                                 .format(datetime.now(), global_step_v,
                                         data_generator.progress))
                    num_samples = 2
                    (inputs_sample_v, inputs_length_sample_v,
                     targets_sample_v, targets_length_sample_v) = (
                        data_generator.sample(num_samples))
                    generated_v = sess.run(
                        fetches=generated,
                        feed_dict={inputs: inputs_sample_v,
                                   inputs_length: inputs_length_sample_v})
                    for i in range(num_samples):
                        logging.info('-' * 60)
                        logging.info('Sample #{}'.format(i))
                        inputs_sample_words = data_generator.ids_to_words(
                            inputs_sample_v[i][:inputs_length_sample_v[i]])
                        targets_sample_words = data_generator.ids_to_words(
                            targets_sample_v[i][1:targets_length_sample_v[i]])
                        generated_words = data_generator.ids_to_words(
                            generated_v[i])
                        if '<EOS>' in generated_words:
                            eos_index = generated_words.index('<EOS>')
                            generated_words = generated_words[:eos_index + 1]
                        logging.info('Input: {}'
                                     .format(' '.join(inputs_sample_words)))
                        logging.info('Target: {}'
                                     .format(' '.join(targets_sample_words)))
                        logging.info('Generated: {}'
                                     .format(' '.join(generated_words)))
                    logging.info('-' * 60)

                if global_step_v % 500 == 0:
                    save_path = os.path.join(save_dir, 'model.ckpt')
                    real_save_path = saver.save(sess=sess, save_path=save_path,
                                                global_step=global_step_v)
                    logging.info('Saved the checkpoint to: {}'
                                 .format(real_save_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the SDAE sae.')
    parser.add_argument('--data', required=True,
                        help='The path of a data file')
    parser.add_argument('--vocab', required=True,
                        help='The path of a vocabulary file')
    parser.add_argument('--save-dir', required=True,
                        help='The path to save sae files')
    parser.add_argument('--word-dim', type=int, default=None,
                        help='The dimension of a word representation')
    parser.add_argument('--sentence-dim', type=int, default=None,
                        help='The dimension of a sentence representation')
    parser.add_argument('--omit-prob', type=float, default=None,
                        help='A probability of a word to be omitted')
    parser.add_argument('--swap-prob', type=float, default=None,
                        help='A probability of adjacent two words to be '
                             'swapped')
    parser.add_argument('--config', default=None,
                        help='The path of a model configuration file')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='The size of a mini-batch')
    parser.add_argument('--max-epoch', type=int, default=5,
                        help='The maximum epoch number')
    parser.add_argument('--max-length', type=int, default=50,
                        help='The maximum number of words; sentences '
                             'longer than this number are ignored')
    args = parser.parse_args()
    main()
