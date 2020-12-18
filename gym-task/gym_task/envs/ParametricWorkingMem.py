import numpy as np
import cv2

import gym
from gym import error, spaces, utils
from gym.utils import seeding

from gym_task.envs.task_env import TaskEnv

# Task 5: Monkey see flashing ligh, Monkey see another flashing light later
#         Monkey decide if the two flashing frequencies match

class ParametricWorkingMem(TaskEnv, gym.Env): 

    def __init__(self):

        super().__init__() 
        self.reset()
        self.action_space = spaces.Discrete(9) 


    def step(self, action):
        return self._categoryStep(action)

    def reset(self):

        self.TIME = 200
        self.experiment = np.zeros((self.TIME, self.DIM_Y, self.DIM_X))

        rate1 = 15 + int(np.random.random_sample() * 10)

        makeRatesSame = np.random.random_sample() < .5
        if makeRatesSame:
            rate2 = rate1
        else:
            if np.random.random_sample() < .5:
                rate2 = rate1+5 + int(np.random.random_sample() * 6)
            else:
                rate2 = rate1-5 - int(np.random.random_sample() * 6)

        halfSec = self.TIME // 8
        flashDuration1 = halfSec * 2 // (rate1 * 2 - 1)
        flashDuration2 = halfSec * 2 // (rate2 * 2 - 1)
        for i in range(0, rate1*2-1, 2):
            self.experiment[halfSec+i*flashDuration1:halfSec+(i+1)*flashDuration1, \
                            self.DIM_X//3:self.DIM_X//3*2, \
                            self.DIM_Y//3:self.DIM_Y//3*2] = 1.

        i_dummy = 0
        for i in range(0, rate2*2-1, 2):
            self.experiment[halfSec*5+i*flashDuration2:halfSec*5+(i+1)*flashDuration2, \
                            self.DIM_X//3:self.DIM_X//3*2, \
                            self.DIM_Y//3:self.DIM_Y//3*2] = 1.
            i_dummy = i

        self.stimulusTime = (halfSec, halfSec*5+(i_dummy+1)*flashDuration2)
        self.fixationTime = (0, self.stimulusTime[1])

        self.experiment[self.fixationTime[0]:self.fixationTime[1], self.midY-1:self.midY+2, self.midX-1:self.midX+2] = 1.

        # 8 > aim >= 4 means they are the same, 4 > aim means they are different (action=8 is waiting)
        self.aim = 7 if rate1 == rate2 else 0 
        self.currFrame = 0

        return self.experiment