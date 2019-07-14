from unittest import main, TestCase
import tensorflow as tf
from consumers.margin_loss import MarginLoss
import numpy as np

BATCH = 5
EMB_SIZE = 10


class TestMarginLoss(TestCase):

    def _fake_tensors(self, support, query):
        return {
            'support_embedding': tf.random_uniform((BATCH, support, EMB_SIZE)),
            'query_embedding': tf.random_uniform((BATCH, query, EMB_SIZE)),
            'prefix': 'test'
        }

    def _check_loss_and_accuracy(self, support, query, scope):
        ml = MarginLoss(0.1)
        with tf.variable_scope(scope):
            outputs = ml.consume(self._fake_tensors(support, query))
            sess = tf.InteractiveSession()
            loss, accuracy, sentences = sess.run(
                [outputs['loss_embedding'], outputs['embedding_accuracy'],
                 outputs['sentences']])
        self.assertIs(type(loss), np.float32)
        self.assertIs(type(accuracy), np.float32)
        self.assertTrue(0.0 <= accuracy <= 1.0)
        self.assertEqual(sentences.shape, (BATCH, EMB_SIZE))

    def test_equal_support_query_size(self):
        self._check_loss_and_accuracy(3, 3, 'test_equal_support_query_size')

    def test_support_more_than_query_size(self):
        self._check_loss_and_accuracy(5, 3, 'test_support_more_than_query_size')

    def test_query_more_than_support_size(self):
        self._check_loss_and_accuracy(3, 5, 'test_query_more_than_support_size')

if __name__ == '__main__':
    main()