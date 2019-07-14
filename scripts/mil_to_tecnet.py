"""Converts the MIL data to a format suited for TecNets"""

import os
import argparse
from multiprocessing import Pool
import pickle
from PIL import Image
from natsort import natsorted

parser = argparse.ArgumentParser()
parser.add_argument('source_dir')
parser.add_argument('target_dir')
parser.add_argument('dataset', choices=['reach', 'push'])
args = parser.parse_args()


def process_task_folder(tasks):
    src_task_path = os.path.join(args.source_dir, tasks)
    if not os.path.isdir(src_task_path):
        return
    task_id = tasks.split('_')[1]
    target_task_path = os.path.join(args.target_dir, 'task_' + task_id)

    task_pkl_opt1 = os.path.join(args.source_dir, task_id + '.pkl')
    task_pkl_opt2 = os.path.join(args.source_dir, 'demos_' + task_id + '.pkl')
    task_pkl = task_pkl_opt1 if os.path.exists(task_pkl_opt1) else task_pkl_opt2

    gif_dirs = natsorted(os.listdir(src_task_path))
    with open(task_pkl, 'rb') as pkl:
        data = pickle.load(pkl, encoding='bytes')
    if args.dataset == 'reach':
        states = data[b'demoX']
        actions = data[b'demoU']
        demo_selection = data[b'demoConditions']
    elif args.dataset == 'push':
        # Mil only used part of the data.
        states = data['demoX'][6:-6]
        actions = data['demoU'][6:-6]
        demo_selection = data['xml']
        gif_dirs = gif_dirs[6:-6]
    else:
        raise RuntimeError('Unrecognized dataset', args.dataset)
    new_data = {
        'states': states,
        'actions': actions,
        'demo_selection': demo_selection
    }

    for gif_file in gif_dirs:
        new_gif_folder = os.path.join(target_task_path, gif_file[:-4])
        print('Splitting gif', gif_file)
        os.makedirs(new_gif_folder)
        with Image.open(os.path.join(src_task_path, gif_file)) as frame:
            nframes = 0
            while frame:
                gif_file_path = os.path.join(new_gif_folder, "%i.gif" % nframes)
                if not os.path.exists(gif_file_path):
                    f = frame.convert("RGB")
                    f.save(gif_file_path, 'gif')
                nframes += 1
                try:
                    frame.seek(nframes)
                except EOFError:
                    break

    new_pickle_path = target_task_path + '.pkl'
    print('Saving', new_pickle_path)
    with open(new_pickle_path, 'wb') as f:
        pickle.dump(new_data, f)


with Pool(20) as p:
   p.map(process_task_folder, os.listdir(args.source_dir))
