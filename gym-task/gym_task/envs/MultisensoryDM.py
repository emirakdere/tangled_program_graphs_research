import numpy as np
import cv2

import gym
from gym import error, spaces, utils
from gym.utils import seeding

from gym_task.envs.task_env import TaskEnv

# Task 4: Mouse hear tick sound or flashing light (inclusive or)
#         Mouse decide whether frequency is same as previously trained frequency
#         In out implementation, right part of the screen represents auditory
#         Left represents visual

class MultisensoryDM(TaskEnv, gym.Env): 

    def __init__(self):

        super().__init__() 
        self.reset()

        # Normally binary decision

        # 8 > action >= 4 <--> matching frequencies
        # 4 > action <--> different frequencies
        # action == 8 <--> wait


    def step(self, action):
        return self._categoryStep(action)

    def reset(self):

        self.TIME = 200
        self.experiment = np.zeros((self.TIME, self.DIM_Y, self.DIM_X))

        freqMouseIsTrainedOn = 20

        makeRateSameAsThreshold = np.random.random_sample() < .5
        if makeRateSameAsThreshold:
            rate = freqMouseIsTrainedOn
        else:
            if np.random.random_sample() < .5:
                rate = freqMouseIsTrainedOn+5 + int(np.random.random_sample() * 6)
            else:
                rate = freqMouseIsTrainedOn-5 - int(np.random.random_sample() * 6)


        halfSec = self.TIME // 8
        flashDuration = halfSec * 4 // (rate * 2 - 1)

        experimentMode = np.random.random_sample()
        doVisual   = 0<experimentMode<.66
        doAuditory = .33<experimentMode<1.

        i_dummy = 0
        for i in range(0, rate*2-1, 2):
            i_dummy = i

            if doVisual:
                self.experiment[halfSec+i*flashDuration:halfSec+(i+1)*flashDuration, \
                                self.DIM_Y//3:self.DIM_Y//3*2,\
                                self.DIM_X//8:self.DIM_X//8*3] = 1.
            if doAuditory:
                self.experiment[halfSec+i*flashDuration:halfSec+(i+1)*flashDuration, \
                                self.DIM_Y//3:self.DIM_Y//3*2, \
                                self.DIM_X//8*5:self.DIM_X//8*7] = 1.

        self.stimulusTime = (halfSec, halfSec+(i_dummy+1)*flashDuration)
        self.fixationTime = (0, self.stimulusTime[1])

        self.experiment[self.fixationTime[0]:self.fixationTime[1], self.midY-1:self.midY+2, self.midX-1:self.midX+2] = 1.

        self.aim = 7 if makeRateSameAsThreshold else 0 

        self.currFrame = 0
        return self.experiment[0]
