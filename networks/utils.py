import tensorflow as tf


def activation(fn_name):
    fn = None
    if fn_name == 'relu':
        fn = tf.nn.relu
    elif fn_name == 'elu':
        fn = tf.nn.elu
    elif fn_name == 'leaky_relu':
        fn = tf.nn.leaky_relu
    return fn
