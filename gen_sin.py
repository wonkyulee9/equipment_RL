# Generator with sine-based function with gaussian noise added and multiplied
# With random offset function

import numpy as np
import random

class gen_sin(core.Env):
    def __init__(self):
        self.offset_1 = 0
        self.time = 0
        self.action_space = [-0.1, 0, 0.1]
        # self.offset_2 = 0

    def generate(self, offset=0):
        self.offset_1 += offset
        g = (np.random.randn(1)/4+1)*np.sin(4*self.time)/8 + np.random.normal(0, 0.66, 1)/5 + self.offset_1
        self.time += 0.4
        return g
    
    def generate_50(self, offset=0, delay=0, inc=0, cont=0):
        obs = np.empty([50, ])
        for x in range(50):
            if x - delay == 0:
                self.offset_add(offset)
            if x - 38 == 0:
                self.offset_add(cont)
            obs[x] = self.generate(inc)
        return obs

    def offset_add(self, offset):
        self.offset_1 += offset
        return
        
    def step(self, action=1):
        done = False
        obs = self.generate_50(self.action_space[action], 38)
        # Reward 정의 - 0 ~ 1 사이의 숫자
        r1 = (obs < 0.5) & (obs > -0.5)
        r2 = (obs < 0.3) & (obs > -0.3)
        reward = (r1.sum() + r2.sum() + 5 * (action == 1))/105
        if reward <= 0.1:
            done = True
        return obs, reward, done

    def step_rand(self, action=1):
        done = False

        off = (np.arange(7)-3)/10
        delay = np.arange(50)
        inc = (2/1000)*(np.arange(7)-3)

        off = np.random.choice(off)
        delay = np.random.choice(delay)
        inc = np.random.choice(inc)
        print(off, delay, inc)
        obs = self.generate_50(off, delay, inc, self.action_space[action])
        r1 = (obs < 0.5) & (obs > -0.5)
        r2 = (obs < 0.3) & (obs > -0.3)
        reward = (r1.sum() + r2.sum() + 5 * (action == 0))/105
        
        if reward <= 0.1:
            done = True
        return obs, reward, done
