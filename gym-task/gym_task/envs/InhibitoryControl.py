import numpy as np
import cv2

import gym
from gym import error, spaces, utils
from gym.utils import seeding

from gym_task.envs.task_env import TaskEnv

# Task 6: Monkey fixate on center fixation, Monkey see stimulus in target location,
#         Monkey see color cue (fixation's color changes),
#         Monkey move towards or away from target based on that color.

class InhibitoryControl(TaskEnv, gym.Env): 

    def __init__(self):

        super().__init__() 
        self.reset()
        self.action_space = spaces.Discrete(9) 

    def step(self, action):
        return _categoryStep(action)

    def reset(self):

        self.TIME = 120
        self.experiment = np.zeros((self.TIME, self.DIM_Y, self.DIM_X))

        halfSec = self.TIME//8
        self.fixationTime = (0, 5*halfSec)
        self.stimulusTime = (2*halfSec, self.TIME)
        colorCue          = (5*halfSec, self.TIME)

        antisaccade = np.random.random_sample() < .5
        targetLoc = [self.midY, 5*(self.DIM_X//6)]

        self.experiment[self.fixationTime[0]:self.fixationTime[1], self.midY-1:self.midY+2, self.midX-1:self.midX+2] = 1.
        self.experiment[self.stimulusTime[0]:self.stimulusTime[1], targetLoc[0]-3:targetLoc[0]+4, targetLoc[1]-3:targetLoc[1]+4] = 1.
        self.experiment[colorCue[0]:colorCue[1], self.midY-1:self.midY+2, self.midX-1:self.midX+2] = .25 if antisaccade else .75

        # 8 > aim >= 4 means antisaccade, 4 > aim means saccade (action=8 is waiting)
        self.aim = 7 if antisaccade else 0 
        self.currFrame = 0
        return self.experiment[0]
