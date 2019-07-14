from unittest import main, TestCase
import tensorflow as tf
from consumers.imitation_loss import ImitationLoss
import numpy as np

BATCH = 5
ACTION_SIZE = 10


class TestImitationLoss(TestCase):

    def _fake_tensors(self):
        return {
            'output_actions': tf.random_uniform((BATCH, 2, ACTION_SIZE)),
            'ctrnet_outputs': tf.random_uniform((BATCH, 2, ACTION_SIZE)),
        }

    def test_float_outputs(self):
        il = ImitationLoss()
        with tf.variable_scope('test_float_outputs'):
            outputs = il.consume(self._fake_tensors())
            sess = tf.InteractiveSession()
            loss_support, loss_query = sess.run(
                [outputs['loss_support'], outputs['loss_query']])
        self.assertIs(type(loss_support), np.float32)
        self.assertIs(type(loss_query), np.float32)


if __name__ == '__main__':
    main()