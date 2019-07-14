from consumers.consumer import Consumer
from networks.input_output import *
import tensorflow as tf

VALID_FRAME_COLLAPSE = ['concat']


class TaskEmbedding(Consumer):

    def __init__(self, network, embedding_size, frame_collapse_method='concat',
                 include_state=False, include_action=False):
        self.network = network
        self.embedding_size = embedding_size
        self.include_state = include_state
        self.include_action = include_action
        if frame_collapse_method not in VALID_FRAME_COLLAPSE:
            raise ValueError('%s is not a valid frame collapse method.'
                             % frame_collapse_method)
        self.frame_collapse_method = frame_collapse_method

    def _squash_input(self, tensor, shape, batch_size, support_plus_query):
        return tf.reshape(tensor, tf.concat(
            [[batch_size * support_plus_query], shape[2:]], axis=0))

    def _expand_output(self, tensor, batch_size,support_plus_query):
        return tf.reshape(tensor, (batch_size, support_plus_query, -1))

    def consume(self, inputs):

        # Condense the inputs from (Batch, support_query, transis, w, h, c)
        # to (Batch, support_query, w, h, c * transis)
        embed_images = self.get(inputs, 'embnet_images')
        support = self.get(inputs, 'support')
        query = self.get(inputs, 'query')

        if self.frame_collapse_method == 'concat':
            embed_images = tf.concat(tf.unstack(embed_images, axis=2), axis=-1)

        embed_images_shape = tf.shape(embed_images)
        batch_size = embed_images_shape[0]
        support_plus_query = embed_images_shape[1]

        # Sanity check
        assertion_op = tf.assert_equal(
            support_plus_query, support + query,
            message='Support and Query size is different than expected.')
        with tf.control_dependencies([assertion_op]):
            # Condense to shape (batch_size*(support_plus_query),w,h,c)
            reshaped_images = self._squash_input(
                embed_images, embed_images_shape, batch_size,
                support_plus_query)
        net_ins = [NetworkInput(name='embed_images', layer_type='conv',
                                layer_num=0, tensor=reshaped_images)]

        if self.include_state:
            embnet_states = self.get(inputs, 'embnet_states')
            if self.frame_collapse_method == 'concat':
                # (Batch, support_query, State*Frames)
                embnet_states = tf.concat(
                    tf.unstack(embnet_states, axis=2), axis=-1)
            reshaped_state = self._squash_input(
                embnet_states, tf.shape(embnet_states), batch_size,
                support_plus_query)
            net_ins.append(NetworkInput(
                name='embnet_states', layer_type='fc',
                layer_num=0, tensor=reshaped_state, merge_mode='concat'))

        if self.include_action:
            embnet_actions = self.get(inputs, 'embnet_outputs')
            if self.frame_collapse_method == 'concat':
                # (Batch, support_query, Actions*Frames)
                embnet_actions = tf.concat(
                    tf.unstack(embnet_actions, axis=2), axis=-1)
            reshaped_action = self._squash_input(
                embnet_actions, tf.shape(embnet_actions), batch_size,
                support_plus_query)
            net_ins.append(NetworkInput(
                name='embnet_actions', layer_type='fc',
                layer_num=0, tensor=reshaped_action, merge_mode='concat'))

        net_out = NetworkHead(name='output_embedding',
                              nodes=self.embedding_size)
        with tf.variable_scope('task_embedding_net', reuse=tf.AUTO_REUSE):
            outputs = self.network.forward(net_ins, [net_out],
                                           self.get(inputs, 'training'))

        # Convert to (Batch, support_query, emb_size)
        embedding = tf.reshape(
            self.get(outputs, 'output_embedding'),
            (batch_size, support_plus_query, self.embedding_size))

        outputs['support_embedding'] = embedding[:, :support]
        outputs['query_embedding'] = embedding[:, support:]

        inputs.update(outputs)
        return inputs
