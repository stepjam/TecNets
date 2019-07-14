from unittest import main, TestCase
import tensorflow as tf
import numpy as np
from networks.cnn import CNN
from networks.input_output import *


class TestNetwork(TestCase):
    """
    This will test is everything to do with the network works configures correctly
    """

    def _inputs(self):
        action = NetworkHead('action', 3, None)
        pose = NetworkHead('pose', 2, 'relu')
        img_in = NetworkInput('img_img', 'conv', 0,
                              tf.placeholder(tf.float32, (None, 8, 8, 3)))
        angles_in = NetworkInput('angles', 'fc', 0,
                                 tf.placeholder(tf.float32, (None, 2)),
                                 'concat', axis=-1)
        return action, pose, img_in, angles_in

    def _basic_cnn(self):
        action, pose, img_in, angles_in = self._inputs()
        net = CNN(filters=[8, 16], fc_layers=[20, 20], kernel_sizes=[3, 3],
                  strides=[2, 2], max_pool=False, norm=None, activation='relu')
        outputs = net.forward([img_in, angles_in], [action, pose],
                              training=None)
        return outputs, img_in, angles_in

    def _norm_cnn(self, norm, scope):
        action, pose, img_in, angles_in = self._inputs()
        net = CNN(filters=[8, 16], fc_layers=[20, 20], kernel_sizes=[3, 3],
                  strides=[2, 2], max_pool=False, norm=norm,
                  activation='relu')

        with tf.variable_scope(scope):
            train = tf.placeholder(tf.bool)
            outputs = net.forward([img_in, angles_in], [action, pose],
                                  training=train)
            sess = tf.InteractiveSession()
            tf.global_variables_initializer().run()
            action, pose = sess.run(
                [outputs['action'], outputs['pose']], feed_dict={
                    img_in.tensor: np.ones((1, 8, 8, 3)),
                    angles_in.tensor: np.ones((1, 2)),
                    train: True
                })
        self.assertEqual(action.shape, (1, 3,))
        self.assertEqual(pose.shape, (1, 2,))

    def test_construct_model(self):
        with tf.variable_scope('test_construct_model'):
            outputs, _, _ = self._basic_cnn()
        self.assertTrue('action' in outputs)
        self.assertTrue('pose' in outputs)

    def test_model_forward_pass(self):
        with tf.variable_scope('test_model_forward_pass'):
            outputs, img_in, angles_in = self._basic_cnn()
            sess = tf.InteractiveSession()
            tf.global_variables_initializer().run()
            action, pose = sess.run(
                [outputs['action'], outputs['pose']], feed_dict={
                    img_in.tensor: np.ones((1, 8, 8, 3)),
                    angles_in.tensor: np.ones((1, 2))
                })
        self.assertEqual(action.shape, (1, 3,))
        self.assertEqual(pose.shape, (1, 2,))

    def test_model_with_batchnorm(self):
        self._norm_cnn('batch', 'test_model_with_batchnorm')

    def test_model_with_layernorm(self):
        self._norm_cnn('layer', 'test_model_with_layernorm')


if __name__ == '__main__':
    main()