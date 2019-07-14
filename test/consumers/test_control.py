from unittest import main, TestCase
import tensorflow as tf
from consumers.control import Control
from networks.cnn import CNN

ACTION_SIZE = 4
STATE_SIZE = 6
BATCH = 5
FRAMES = 3
EMB_SIZE = 7


class TestControl(TestCase):

    def _fake_tensors(self):
        return {
            'sentences': tf.random_uniform((BATCH, EMB_SIZE)),
            'ctrnet_images': tf.random_uniform((BATCH, 2, 32, 32, 3)),
            'ctrnet_states': tf.random_uniform((BATCH, 2, STATE_SIZE)),
            'training': True
        }

    def _check_loss(self, scope, state):
        net = CNN(filters=[8, 16], fc_layers=[20, 20], kernel_sizes=[3, 3],
                 strides=[2, 2], max_pool=False, norm=None, activation='relu')
        c = Control(network=net, action_size=ACTION_SIZE, include_state=state)
        with tf.variable_scope(scope):
            outputs = c.consume(self._fake_tensors())
            sess = tf.InteractiveSession()
            tf.global_variables_initializer().run()
            actions = sess.run(outputs['output_actions'])
        self.assertEqual(actions.shape, (BATCH, 2, ACTION_SIZE))

    def test_no_state(self):
        self._check_loss('test_no_state', False)

    def test_with_state(self):
        self._check_loss('test_with_state', True)


if __name__ == '__main__':
    main()