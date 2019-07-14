from consumers.consumer import Consumer
import tensorflow as tf
from data import utils


class EvalConsumer(Consumer):

    def __init__(self, dataset, data_sequencer, support, disk_images=True):
        self.dataset = dataset
        self.data_sequencer = data_sequencer
        self.support = support
        self.disk_images = disk_images
        super().__init__()

    def consume(self, inputs):

        if self.disk_images:
            # (Examples,)
            input_image = tf.placeholder(tf.string, (self.support,))
        else:
            # (Examples, timesteps)
            input_image = tf.placeholder(tf.float32,
                                         (None, None) + self.dataset.img_shape)
        input_states = tf.placeholder(
            tf.float32,
            (self.support, self.dataset.time_horizon, self.dataset.state_size))
        input_outputs = tf.placeholder(
            tf.float32,
            (self.support, self.dataset.time_horizon, self.dataset.action_size))

        # (B. W, H, C)
        input_ctr_image = tf.placeholder(tf.float32,
                                         (None, 1) + self.dataset.img_shape)
        input_ctr_state = tf.placeholder(tf.float32,
                                         (None, 1, self.dataset.state_size))

        training = tf.placeholder_with_default(False, None)

        stacked_embnet_images, bs, cs = [], [], []
        for i in range(self.support):
            embnet_images, embnet_states, embnet_outputs = (
                self.data_sequencer.load(
                    input_image[i], input_states[i], input_outputs[i]))
            embnet_images = utils.preprocess(embnet_images)
            stacked_embnet_images.append(embnet_images)
            bs.append(embnet_states)
            cs.append(embnet_outputs)

        embnet_images = tf.stack(stacked_embnet_images)
        embnet_images = tf.expand_dims(embnet_images, axis=0)  # set batchsize 1

        embnet_states = tf.stack(bs)
        embnet_states = tf.expand_dims(embnet_states, axis=0)

        embnet_outputs = tf.stack(cs)
        embnet_outputs = tf.expand_dims(embnet_outputs, axis=0)

        embnet_images.set_shape(
            (None, None, self.data_sequencer.frames) + self.dataset.img_shape)
        embnet_states.set_shape(
            (None, None, self.data_sequencer.frames, self.dataset.state_size))
        embnet_outputs.set_shape(
            (None, None, self.data_sequencer.frames, self.dataset.action_size))

        return {
            'embnet_images': embnet_images,
            'embnet_states': embnet_states,
            'embnet_outputs': embnet_outputs,
            'input_image_files': input_image,
            'input_states': input_states,
            'input_outputs': input_outputs,
            'ctrnet_images': input_ctr_image,
            'ctrnet_states': input_ctr_state,
            'training': training,
            'support': tf.placeholder_with_default(self.support, None),
            'query': tf.placeholder_with_default(0, None),
        }
