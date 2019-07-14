from unittest import main, TestCase
import tensorflow as tf
from data.data_sequencer import DataSequencer
import numpy as np

TIME = 10
STATES = 2
OUTS = 3


class TestDataSequencer(TestCase):

    def _ram_data(self, sequence_strategy, sequence_num, scope):
        ds = DataSequencer(sequence_strategy, TIME)
        with tf.variable_scope(scope):
            images = tf.random_uniform((TIME, 8, 8, 3))
            states = tf.random_uniform((TIME, STATES))
            outputs = tf.random_uniform((TIME, OUTS))
            limgs, lstates, louts = ds.load(images, states, outputs)
            sess = tf.InteractiveSession()
            out_imgs, out_states, out_outs = sess.run([limgs, lstates, louts])
            self.assertEqual(np.array(out_imgs).shape, (sequence_num, 8, 8, 3))
            self.assertEqual(np.array(out_states).shape, (sequence_num, STATES))
            self.assertEqual(np.array(out_outs).shape, (sequence_num, OUTS))

    def test_ram_first(self):
        self._ram_data('first', 1, 'test_ram_first')

    def test_ram_last(self):
        self._ram_data('last', 1, 'test_ram_last')

    def test_ram_first_last(self):
        self._ram_data('first_last', 2, 'test_ram_first_last')

    def test_ram_all(self):
        self._ram_data('all', TIME, 'test_ram_all')


if __name__ == '__main__':
    main()