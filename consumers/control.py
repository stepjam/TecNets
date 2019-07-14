from consumers.consumer import Consumer
import tensorflow as tf
from networks.input_output import *


class Control(Consumer):

    def __init__(self, network, action_size, include_state=False):
        self.network = network
        self.action_size = action_size
        self.include_state = include_state
        super().__init__()

    def consume(self, inputs):

        # Shape (batch, embsize)
        s = self.get(inputs, 'sentences')

        # (batch, examples, h, w, 3)
        ctrnet_images = self.get(inputs, 'ctrnet_images')

        examples = ctrnet_images.shape[1]
        width = ctrnet_images.shape[-2]
        height = ctrnet_images.shape[-3]

        # (batch, 1, 1, 1, emb)
        s = tf.expand_dims(
            tf.expand_dims(tf.expand_dims(s, axis=1), axis=1), axis=1)
        # (batch, examples, height, width, emb)
        tiled = tf.tile(s, [1, examples, height, width, 1])
        ctrnet_input = tf.concat([ctrnet_images, tiled], axis=-1)
        emb_plus_channels = ctrnet_input.shape[-1]

        # Squash (batch * examples, h, w, emb)
        ctrnet_input = tf.reshape(
            ctrnet_input, (-1, height, width, emb_plus_channels))

        net_ins = [NetworkInput(name='ctr_images', layer_type='conv',
                                layer_num=0, tensor=ctrnet_input)]

        if self.include_state:
            states = self.get(inputs, 'ctrnet_states')
            states = tf.reshape(states, (-1, states.shape[-1]))
            net_ins.append(NetworkInput(
                name='ctrnet_states', layer_type='fc',
                layer_num=0, tensor=states, merge_mode='concat'))

        net_out = NetworkHead(name='output_action',
                              nodes=self.action_size)

        with tf.variable_scope('control_net', reuse=tf.AUTO_REUSE):
            outputs = self.network.forward(net_ins, [net_out],
                                           self.get(inputs, 'training'))

        inputs['output_actions'] = tf.reshape(self.get(outputs, 'output_action'),
                                           (-1, examples, self.action_size))
        return inputs
