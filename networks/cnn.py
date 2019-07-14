import tensorflow as tf
from networks.utils import *
from networks.input_output import *


class CNN(object):

    def __init__(self, filters, fc_layers, kernel_sizes, strides=None,
                 max_pool=False, drop_rate=0.0, norm=None, activation='relu'):
        """Initializes a standard CNN network.

        :param filters: List of number of filters per convolution layer.
        :param fc_layers: List of fully connected units per layer.
        :param normalization: String defining the type of normalization.
        """
        self.filters = filters
        self.fc_layers = fc_layers
        self.norm = norm
        self.drop_rate = drop_rate
        self.kernel_sizes = kernel_sizes
        self.activation = activation
        self.max_pool = max_pool
        self.strides = strides if strides is not None else [1] * len(fc_layers)
        if not max_pool and strides is None:
            raise RuntimeError('No dimensionality reduction.')

    def _pre_layer_util(self, layer, cur_layer_num, ins):
        for cin in ins:
            if cin.layer_num > cur_layer_num:
                break
            elif cin.layer_num == cur_layer_num:
                if cin.merge_mode == 'concat':
                    layer = tf.concat([layer, cin.tensor], axis=cin.axis)
                elif cin.merge_mode == 'addition':
                    layer += cin.tensor
                elif cin.merge_mode == 'multiply':
                    layer *= cin.tensor
                else:
                    raise RuntimeError('Unrecognised merging method for %s.' %
                                       cin.name)
        return layer

    def _post_layer_util(self, layer, training, norm):

        if self.drop_rate > 0:
            layer = tf.layers.dropout(layer, rate=0.5, training=training)

        act_fn = activation(self.activation)
        if norm and self.norm is not None:
            if self.norm == 'batch':
                layer = tf.contrib.layers.batch_norm(
                    layer, is_training=training, activation_fn=act_fn)
            elif self.norm == 'layer':
                layer = tf.contrib.layers.layer_norm(
                    layer, activation_fn=act_fn)
            else:
                raise RuntimeError('Unsupported normalization method: %s'
                                   % self.norm)
        else:
            layer = act_fn(layer)
        return layer

    def forward(self, inputs, heads, training):
        """Inputs want to be fused in at different times. """

        inputs = sorted(inputs, key=lambda item: item.layer_num)
        conv_inputs = list(filter(lambda item: item.layer_type == 'conv', inputs))
        fc_inputs = list(filter(lambda item: item.layer_type == 'fc', inputs))

        if conv_inputs[0].layer_num > 0:
            raise RuntimeError('Need an input tensor.')
        elif len(conv_inputs) > 1 and conv_inputs[1].layer_num == 0:
            raise RuntimeError('Can only have one main input tensor.')

        layer = conv_inputs[0].tensor
        del conv_inputs[0]

        outputs = {}

        for i, (filters, ksize, stride) in enumerate(
                zip(self.filters, self.kernel_sizes, self.strides)):
            layer = self._pre_layer_util(layer, i, conv_inputs)
            layer = tf.layers.conv2d(layer, filters, ksize, stride, 'same')
            layer = self._post_layer_util(layer, training, True)

        layer = tf.layers.flatten(layer)
        for i, fc_layers in enumerate(self.fc_layers):
            layer = self._pre_layer_util(layer, i, fc_inputs)
            layer = tf.layers.dense(layer, fc_layers)
            layer = self._post_layer_util(layer, training, False)

        for head in heads:
            act_fn = activation(head.activation)
            output = tf.layers.dense(layer, head.nodes)
            outputs[head.name] = output if act_fn is None else act_fn(output)

        return outputs
