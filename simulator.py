import numpy as np
import random
import os

XSIZE = 50                                          # Generated output size


# Simulator (inherit keras.rl Env class)                            # Data-based Simulator: Use preprocessed data and control-state relations to generate close-to-real data
class gen(core.Env):
    def __init__(self, path):
        self.offset = np.array([0.0, 0.0, 0.0], dtype='float32')
        self.sess = -1
        self.step = 0
        self.action_space = np.array([-.2, -.1, 0, .1, .2])
        self.data = []

        file_list = os.listdir(path)
        self.flist = file_list

        for item in file_list:
            csv_data = np.loadtxt(path + '/' + item, delimiter=',', dtype=np.float32)
            g_real = csv_data[:, :3]
            self.data.append(g_real)
            print("{} load succes from {}".format(item, path))

    def data_shuff(self):                                                           # Shuffle data order
        idx = np.arange(0, len(self.data))
        np.random.shuffle(idx)
        data_shuffle = [self.data[ind] for ind in idx]
        self.data = data_shuffle
        return

    def generate(self, offset=np.array([0, 0, 0], dtype='float32')):                # Generate one point
        self.offset += offset
        g = self.data[self.sess][self.step] + self.offset
        self.step += 1
        return g

    def generate_50(self, offset=np.array([0, 0, 0], dtype='float32')):             # Generate 50 points (1 state)
        obs = np.empty([XSIZE, 3])
        for x in range(XSIZE):
            if x == 40:
                obs[x] = self.generate(offset)
            else:
                obs[x] = self.generate()
        return obs

    def reward(self, n_state, action):                                              # Observe state and reward accordingly
        # Reward #2
        a = np.abs(n_state[40:, 2]) < 0.3
        reward = (a.sum() * (1 + 0.2 * (action == 2)) * (1 + 0.1 * (action == 1 or action == 3))) / 12

        return reward

    def next_state(self, action1=2, action2=2):                                     # Function to return state and reward according to some control actions
        off = np.array(self.action_space[[action1, 4-action1, action2]])
        obs = self.generate_50(offset=off)
        reward = self.reward(obs, action2)
        if len(self.data[self.sess]) - self.step < 50:
            done = True
        else:
            done = False
        return obs, reward, done

    def reset(self):                                                                # Reset and move on to next session when one ends
        self.offset = np.array([0.0, 0.0, 0.0], dtype='float32')
        self.sess += 1
        if self.sess >= len(self.data):
            self.sess = 0
        self.step = 0
        obs, _, _ = self.next_state()
        return obs

    def close(self):
        return
