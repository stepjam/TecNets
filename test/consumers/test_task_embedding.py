from unittest import main, TestCase
import tensorflow as tf
from consumers.task_embedding import TaskEmbedding
from networks.cnn import CNN

STATE_SIZE = 3
ACTION_SIZE = 4
BATCH = 5
FRAMES = 3
EMB_SIZE = 7


class TestTaskEmbedding(TestCase):

    def _fake_tensors(self, support, query):
        return {
            'embnet_images': tf.random_uniform(
                (BATCH, support + query, FRAMES, 32, 32, 3)),
            'embnet_states': tf.random_uniform(
                (BATCH, support + query, FRAMES, STATE_SIZE)),
            'embnet_outputs': tf.random_uniform(
                (BATCH, support + query, FRAMES, ACTION_SIZE)),
            'training': False,
            'support': support,
            'query': query
        }

    def _check_loss_and_accuracy(self, support, query, scope, state, action):
        net = CNN(filters=[8, 16], fc_layers=[20, 20], kernel_sizes=[3, 3],
                 strides=[2, 2], max_pool=False, norm=None, activation='relu')
        te = TaskEmbedding(network=net, embedding_size=EMB_SIZE,
                           frame_collapse_method='concat',
                           include_state=state, include_action=action)
        with tf.variable_scope(scope):
            outputs = te.consume(self._fake_tensors(support, query))
            sess = tf.InteractiveSession()
            tf.global_variables_initializer().run()
            sup_emb, que_emb = sess.run(
                [outputs['support_embedding'], outputs['query_embedding']])
        self.assertEqual(sup_emb.shape, (BATCH, support, EMB_SIZE))
        self.assertEqual(que_emb.shape, (BATCH, support, EMB_SIZE))

    def test_no_state_action(self):
        self._check_loss_and_accuracy(
            3, 3, 'test_no_state_action', False, False)

    def test_with_state(self):
        self._check_loss_and_accuracy(
            3, 3, 'test_with_state', True, False)

    def test_with_state_and_action(self):
        self._check_loss_and_accuracy(
            3, 3, 'test_with_state_and_action', True, True)


if __name__ == '__main__':
    main()