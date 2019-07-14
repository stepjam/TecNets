from data import utils
import tensorflow as tf

VALID_SEQUENCE_STRATEGIES = ['first', 'last', 'first_last', 'all']


class DataSequencer(object):

    def __init__(self, sequence_strategy, time_horizon):
        self.sequence_strategy = sequence_strategy
        self.time_horizon = time_horizon
        if sequence_strategy not in VALID_SEQUENCE_STRATEGIES:
            raise ValueError('%s is not a valid sequence embedding strategy.'
                             % sequence_strategy)
        self.frames = 1
        if sequence_strategy == 'first_last':
            self.frames = 2
        elif sequence_strategy == 'all':
            self.frames = self.time_horizon

    def load(self, images, states, outputs):
        is_image_file = images.dtype == tf.string
        # Embedding images
        if self.sequence_strategy == 'first':
            if is_image_file:
                loaded_images = [utils.tf_load_image(images, 0)]
            else:
                loaded_images = [images[0]]
            emb_states = [states[0]]
            emb_outputs = [outputs[0]]
        elif self.sequence_strategy == 'last':
            if is_image_file:
                loaded_images = [utils.tf_load_image(images,
                                              self.time_horizon - 1)]
            else:
                loaded_images = [images[self.time_horizon - 1]]
            emb_states = [states[-1]]
            emb_outputs = [outputs[-1]]
        elif self.sequence_strategy == 'first_last':
            if is_image_file:
                loaded_images = [utils.tf_load_image(images, 0),
                                 utils.tf_load_image(images,
                                                     self.time_horizon - 1)]
            else:
                loaded_images = [images[0], images[self.time_horizon - 1]]
            emb_states = [states[0], states[-1]]
            emb_outputs = [outputs[0], outputs[-1]]
        elif self.sequence_strategy == 'all':
            if is_image_file:
                loaded_images = [utils.tf_load_image(images, t)
                          for t in range(self.time_horizon)]
            else:
                loaded_images = images
            emb_states = [states[t]
                          for t in range(self.time_horizon)]
            emb_outputs = [outputs[t]
                           for t in range(self.time_horizon)]
        else:
            raise ValueError(
                '%s is not a valid sequence embedding strategy.'
                % self.sequence_strategy)
        return loaded_images, emb_states, emb_outputs
