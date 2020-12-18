import numpy as np
import cv2

import gym
from gym import error, spaces, utils
from gym.utils import seeding

from gym_task.envs.task_env import TaskEnv


# Task 8: Monkey watch dots moving in one direction, Monkey rest
#         Monkey watch a second group of dots moving in another direction
#         Monkey decide if these are in the same predefined category

class DelayedMatchToCategory(TaskEnv, gym.Env):

    def __init__(self):

        super().__init__() 
        self.reset()
        

    def step(self, action):
        return self._categoryStep(action)

    def reset(self):

        self.TIME = 120
        self.experiment = np.zeros((self.TIME, self.DIM_Y, self.DIM_X))


        halfSecond = self.TIME // 8
        stimulusTime = (halfSecond, 3*halfSecond)
        secondStimulusTime = (5*halfSecond, 7*halfSecond)
        self.stimulusTime = (stimulusTime[0], secondStimulusTime[1])
        self.fixationTime = (0, self.stimulusTime[1])

        # angles defining the categories
        lower = 3/4*np.pi
        upper = 7/4*np.pi

        randomAngle1 = np.random.random_sample() * 2 * np.pi
        randomAngle2 = np.random.random_sample() * 2 * np.pi

        # see if they are equal to see if they are matching categories
        categoryOf1  = lower < randomAngle1 < upper
        categoryOf2  = lower < randomAngle2 < upper

        self._dotExperiment(1., 30, stimulusTime, dotColor=1., angle=randomAngle1)
        self._dotExperiment(1., 30, secondStimulusTime, dotColor=1., angle=randomAngle2)

        self.experiment[self.fixationTime[0]:self.fixationTime[1], self.midY-1:self.midY+2, self.midX-1:self.midX+2] = 1.


        self.aim = 7 if categoryOf1 == categoryOf2 else 0
        self.currFrame = 0
        return self.experiment[0]
