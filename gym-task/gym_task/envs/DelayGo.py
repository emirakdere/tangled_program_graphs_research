import numpy as np
import cv2

import gym
from gym import error, spaces, utils
from gym.utils import seeding

from gym_task.envs.task_env import TaskEnv

# Task 1: memory guided response; Monkey focus, monkey see stimulus, 
#         focus vanish, monkey saccade.

class DelayGo(TaskEnv, gym.Env): 

    def __init__(self):

        super().__init__() 
        self.reset()

        # TODO: normally continuous (saccade with 6 degrees around the cue location is considered correct)
        #       Right now, assume discrete for simplicity

        # action \in {0, 1, ..., 7} <--> saccade to the location action * 45 degrees
        # action == 8 <--> wait

    def step(self, action):
        return self._continuousStep(action)

    def reset(self):
        self.TIME = 10
        self.experiment = np.zeros((self.TIME, self.DIM_Y, self.DIM_X))

        self.fixationTime = (0, self.TIME-3)
        self.stimulusTime = (2, 4)

        self.experiment[self.fixationTime[0]:self.fixationTime[1], self.midY-1:self.midY+2, self.midX-1:self.midX+2] = 1.    
        stimulusX, stimulusY, randomNumber = self.stimulusPos(.6 * self.DIM_X / 2, 45)
        self.aim = int(randomNumber * 8)

        self.experiment[self.stimulusTime[0]:self.stimulusTime[1], stimulusY-1:stimulusY+2, stimulusX-1:stimulusX+2] = 1.

        self.currFrame = 0
        return self.experiment[0]
