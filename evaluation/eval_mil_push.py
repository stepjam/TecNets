import os
import numpy as np
import random
from gym.envs.mujoco.pusher import PusherEnv
from evaluation.eval import Eval
from data import utils

XML_FOLDER = "/media/stephen/c6c2821e-ed17-493a-b35b-4b66f0b21ee7/MIL/gym/gym/envs/mujoco/assets"


class EvalMilPush(Eval):

    def _load_env(self, xml):
        xml = xml[xml.rfind('pusher'):]
        xml_file = 'sim_push_xmls/test2_ensure_woodtable_distractor_%s' % xml
        xml_file = os.path.join(XML_FOLDER, xml_file)
        env = PusherEnv(**{'xml_file': xml_file, 'distractors': True})
        env.set_visibility(self.render)
        env.render()
        viewer = env.viewer
        viewer.autoscale()
        viewer.cam.trackbodyid = -1
        viewer.cam.lookat[0] = 0.4
        viewer.cam.lookat[1] = -0.1
        viewer.cam.lookat[2] = 0.0
        viewer.cam.distance = 0.75
        viewer.cam.elevation = -50
        viewer.cam.azimuth = -90
        return env

    def _eval_success(self, obs):
        obs = np.array(obs)
        target = obs[:, -3:-1]
        obj = obs[:, -6:-4]
        dists = np.sum((target - obj) ** 2, 1)  # distances at each time step
        return np.sum(dists < 0.017) >= 10

    def evaluate(self, iter):

        print("Evaluating at iteration: %i" % iter)
        iter_dir = os.path.join(self.record_gifs_dir, 'iter_%i' % iter)
        utils.create_dir(iter_dir)

        successes = []
        for i in range(self.num_tasks):

            # demo_selection will be an xml file
            env = self._load_env(self.demos[i][0]['demo_selection'])

            selected_demo_indexs = random.sample(
                range(len(self.demos[i])), self.supports)

            embedding = self.get_embedding(i, selected_demo_indexs)
            gifs_dir = self.create_gif_dir(iter_dir, i)

            for j in range(self.num_trials):
                env.reset()
                observations = []
                world_state = []
                for t in range(self.time_horizon):
                    env.render()
                    # Observation is shape  (100,100,3)
                    obs, state = env.get_current_image_obs()
                    observations.append(obs)
                    obs = ((obs / 255.0) * 2.) - 1.

                    action = self.get_action(obs, state, embedding)
                    ob, reward, done, reward_dict = env.step(np.squeeze(action))
                    world_state.append(np.squeeze(ob))
                    if done:
                        break

                if self._eval_success(world_state):
                    successes.append(1.)
                else:
                    successes.append(0.)
                self.save_gifs(observations, gifs_dir, j)

            env.render(close=True)

        final_suc = np.mean(successes)
        print("Final success rate is %.5f" % (final_suc))
        return final_suc
