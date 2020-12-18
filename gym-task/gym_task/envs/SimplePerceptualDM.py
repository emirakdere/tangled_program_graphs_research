import numpy as np
import cv2

import gym
from gym import error, spaces, utils
from gym.utils import seeding

from gym_task.envs.task_env import TaskEnv

# Task 2: Monkey follows with saccade where all the cohesive dots are going
#         Note that the 80% of the dots are moving in the same direction


class SimplePerceptualDM(TaskEnv, gym.Env): 

    def __init__(self):

        super().__init__() 
        self.reset()

        # TODO: normally binary decision (top vs bottom, right vs left etc...)

        # action \in {0, 1, ..., 7} <--> saccade to the location action * 45 degrees
        # action == 8 <--> wait
    
    def step(self, action):
        return self._continuousStep(action)

    def reset(self):
        
        self.TIME = 10
        self.experiment = np.zeros((self.TIME, self.DIM_Y, self.DIM_X))
        self.stimulusTime = (2, self.TIME) 
        self.fixationTime = (3, 3)
        randomNumber = self._dotExperiment(.80, 20, self.stimulusTime, pieSizeInDegrees=45)        
        self.aim = int(randomNumber * 8)
        self.currFrame = 0
        return self.experiment[0]


