from unittest import main, TestCase
from unittest.mock import MagicMock
from data.generator import Generator
import numpy as np
import tensorflow as tf
import os
from data.data_sequencer import DataSequencer


TIME_HORIZON = 4
BATCH_SIZE = 3
SUPPORT_SIZE = 2
QUERY_SIZE = 2
TASKS = 20
EXAMPLES = 6
IMG_SHAPE = (8, 8, 3)
STATE_SIZE = 11
OUTPUT_SIZE = 9


class TestGenerator(TestCase):

    def _fake_dataset_load(self, tasks, examples):
        fake_folder = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            '../test_data', 'test_task')
        data = [[{
            'image_files': fake_folder,
            'states': np.ones((TIME_HORIZON, STATE_SIZE)),
            'actions': np.ones((TIME_HORIZON, OUTPUT_SIZE))
        } for _ in range(examples)] for _ in range(tasks)]
        # Return fake train and validation data
        return data, data

    def _fake_dataset(self, tasks, examples):
        dataset = MagicMock()
        dataset.time_horizon = TIME_HORIZON
        dataset.training_set = MagicMock(
            return_value=self._fake_dataset_load(tasks, examples))
        return dataset

    # TODO: Should move the data_sequencer code to the correct test class
    def _embedding_strategy(self, scope, strategy, frames,
                            batch_size=BATCH_SIZE, support_size=SUPPORT_SIZE,
                            query_size=QUERY_SIZE):
        dataset = self._fake_dataset(TASKS, EXAMPLES)
        data_seq = DataSequencer(strategy, TIME_HORIZON)
        gen = Generator(dataset, batch_size, support_size, query_size,
                        data_sequencer=data_seq)
        with tf.variable_scope(scope):
            sess = tf.InteractiveSession()
            train_handle, val_handle = gen.get_handles(sess)
            (embed_images, embnet_states, embnet_outputs, ctrnet_images,
             ctrnet_states, ctrnet_outputs) = sess.run(
                gen.next_element, feed_dict={gen.handle: train_handle})
        self.assertEqual(
            embed_images.shape,
            (batch_size, support_size + query_size, frames) + IMG_SHAPE)
        self.assertEqual(
            embnet_states.shape,
            (batch_size, support_size + query_size, frames, STATE_SIZE))
        self.assertEqual(
            embnet_outputs.shape,
            (batch_size, support_size + query_size, frames, OUTPUT_SIZE))
        self.assertEqual(ctrnet_images.shape, (batch_size, 2) + IMG_SHAPE)
        self.assertEqual(ctrnet_states.shape, (batch_size, 2, STATE_SIZE))
        self.assertEqual(ctrnet_outputs.shape, (batch_size, 2, OUTPUT_SIZE))

    def test_first_frame_embedding(self):
        self._embedding_strategy('test_first_frame_embedding', 'first', 1)

    def test_last_frame_embedding(self):
        self._embedding_strategy('test_last_frame_embedding', 'last', 1)

    def test_first_last_frame_embedding(self):
        self._embedding_strategy('test_first_last_frame_embedding',
                                 'first_last', 2)

    def test_all_frame_embedding(self):
        self._embedding_strategy('test_all_frame_embedding', 'all',
                                 TIME_HORIZON)

    def test_invalid_frame_embedding_throws_error(self):
        with self.assertRaises(ValueError):
            self._embedding_strategy(
                'test_invalid_frame_embedding_throws_error', 'invalid', 1)

    def test_support_and_query_more_than_samples(self):
        with self.assertRaises(Exception):
            self._embedding_strategy(
                'test_support_and_query_more_than_samples', 'first',
                1, support_size=TIME_HORIZON+1)


if __name__ == '__main__':
    main()
