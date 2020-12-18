import numpy as np
import cv2

import gym
from gym import error, spaces, utils
from gym.utils import seeding

from gym_task.envs.task_env import TaskEnv

# Task 3: Depending on the given context, monkey either follows the direction 
#         of the cohesive dots (no noise, dots all move in the same direction)
#         or makes a decision based on the color of the dots

class ContextDependentDM(TaskEnv, gym.Env): 


    def __init__(self):

        super().__init__() 
        self.reset()

        # TODO: normally binary decision (top vs bottom, right vs left etc...)
        #       Remove other task variations

        # action \in {0, 1, ..., 7} <--> saccade to the location action * 45 degrees
        # action == 8 <--> wait

    def step(self, action):
        return self._continuousStep(action)

    def reset(self):
        # Fixation (0.5 s) Cue (1s) Stimulus (<3 s) Response
        self.TIME = 20
        self.experiment = np.zeros((self.TIME, self.DIM_Y, self.DIM_X))

        halfSec = self.TIME//8
        self.fixationTime = (0, 6*halfSec)
        self.cueTime = (halfSec, halfSec + 2*halfSec)
        self.stimulusTime = (self.cueTime[1], 6*halfSec)

        cueShapeInMiddle = np.zeros((self.DIM_Y, self.DIM_X))

        moveBasedOnDirection = np.random.random_sample() > .5
        if moveBasedOnDirection:
            self._putToMiddle(cueShapeInMiddle, "X.jpg")
        else:
            self._putToMiddle(cueShapeInMiddle, "O.jpg")

        self.experiment[self.cueTime[0]:self.cueTime[1]] = cueShapeInMiddle

        color = np.random.random_sample()
        randomNumber = self._dotExperiment(1., 50, self.stimulusTime, dotColor=color)
        self.experiment[self.fixationTime[0]:self.fixationTime[1], self.midY-1:self.midY+2, self.midX-1:self.midX+2] = 1.

        
        self.aim = int(randomNumber * 8) if moveBasedOnDirection else int(color * 8)
        self.currFrame = 0
        return self.experiment[0]



