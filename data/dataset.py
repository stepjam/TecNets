import os
import pickle
from natsort import natsorted


class Dataset(object):

    def __init__(self, name, img_shape, state_size, action_size, time_horizon,
                 training_size=None, validation_size=None):
        self.name = name
        self.img_shape = img_shape
        self.state_size = state_size
        self.action_size = action_size
        self.time_horizon = time_horizon
        self.training_size = training_size
        self.validation_size = validation_size
        self.data_root = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), '../datasets', name)

    def training_set(self):
        tasks = self.load('train', self.training_size + self.validation_size)
        return tasks[:self.training_size], tasks[-self.validation_size:]

    def test_set(self):
        return self.load("test")

    def load(self, train_or_test, count=None):
        """Expected to be the test or train folder"""
        train_test_dir = os.path.join(self.data_root, train_or_test)
        tasks = []
        for task_f in natsorted(os.listdir(train_test_dir)):
            task_path = os.path.join(train_test_dir, task_f)
            if not os.path.isdir(task_path):
                continue
            pkl_file = task_path + '.pkl'
            with open(pkl_file, "rb") as f:
                data = pickle.load(f)
            example_img_folders = natsorted(os.listdir(task_path))
            examples = []
            for e_idx, ex_file in enumerate(example_img_folders):
                img_path = os.path.join(task_path, ex_file)
                example = {
                    'image_files': img_path,
                    'actions': data['actions'][e_idx],
                    'states': data['states'][e_idx]
                }
                if 'demo_selection' in data:
                    example['demo_selection'] = data['demo_selection']
                examples.append(example)
            tasks.append(examples)
            if count is not None and len(tasks) >= count:
                break
        return tasks

    def get_outputs(self):
        return {
            'actions': (self.action_size,)
        }

    def get_inputs(self):
        return {
            'states': (self.state_size,)
        }
