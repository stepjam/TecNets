import os
import tensorflow as tf


def create_dir(path):
    exist = os.path.exists(path)
    if not exist:
        os.makedirs(path)
    return exist


def tf_load_image(foldername, timestep):
    file = tf.string_join([foldername, '/', tf.as_string(timestep), '.gif'])
    return tf.image.decode_gif(tf.read_file(file))[0]


def preprocess(img):
    # In range [-1, 1]
    return ((tf.cast(img, tf.float32) / 255.) * 2.) - 1.
