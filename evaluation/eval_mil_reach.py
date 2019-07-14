import numpy as np
import os
import random
import gym
from evaluation.eval import Eval
from data import utils

REACH_SUCCESS_THRESH = 0.05
REACH_SUCCESS_TIME = 10


class EvalMilReach(Eval):

    def __init__(self, sess, dataset, outputs, supports, num_tasks,
                 num_trials, log_dir=".", record_gifs=False, render=True):
        super().__init__(sess, dataset, outputs, supports, num_tasks,
                         num_trials, log_dir, record_gifs, render)
        self.env = gym.make('ReacherMILTest-v1')
        self.env.env.set_visibility(render)

    def evaluate(self, iter):

        print("Evaluating at iteration: %i" % iter)
        iter_dir = os.path.join(self.record_gifs_dir, 'iter_%i' % iter)
        utils.create_dir(iter_dir)
        self.env.reset()

        successes = []
        for i in range(self.num_tasks):

            # TODO hacked in for now. Remove 0
            dem_conds = self.demos[i][0]['demo_selection']

            # randomly select a demo from each of the folders
            selected_demo_indexs = random.sample(
                range(len(dem_conds)), self.supports)

            embedding = self.get_embedding(i, selected_demo_indexs)
            gifs_dir = self.create_gif_dir(iter_dir, i)

            for j in range(self.num_trials):
                if j in dem_conds:
                    distances = []
                    observations = []
                    for t in range(self.time_horizon):
                        self.env.render()
                        # Observation is shape (64,80,3)
                        obs, state = self.env.env.get_current_image_obs()
                        observations.append(obs)
                        obs = ((obs / 255.0) * 2.) - 1.

                        action = self.get_action(obs, state, embedding)
                        ob, reward, done, reward_dict = self.env.step(
                            np.squeeze(action))
                        dist = -reward_dict['reward_dist']
                        if t >= self.time_horizon - REACH_SUCCESS_TIME:
                            distances.append(dist)
                    if np.amin(distances) <= REACH_SUCCESS_THRESH:
                        successes.append(1.)
                    else:
                        successes.append(0.)
                    self.save_gifs(observations, gifs_dir, j)

                self.env.render(close=True)
                self.env.env.next()
                self.env.env.set_visibility(self.render)
                self.env.render()

        self.env.render(close=True)
        self.env.env.reset_iter()
        final_suc = np.mean(successes)
        print("Final success rate is %.5f" % final_suc)
        return final_suc
