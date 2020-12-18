import numpy as np
import cv2
import random

import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym_task.envs.task_env import TaskEnv

# Task 7: Monkey see picture A, Monkey only do something if Monkey see picture A again

# TODO: Implementation details on hold tentatively until step is confirmed
class DelayedMatchToSample(TaskEnv, gym.Env):

    def __init__(self):

        super().__init__()
        self.reset()
        self.action_space = spaces.Discrete(9) 


    def step(self, action):

        return None
        # self.currFrame += 1
        # experimentOver = False if self.currFrame < self.TIME else True
        # reward = 0
        # return self.experiment[self.currFrame-1], reward, experimentOver, {}

    def reset(self):
        # .5 sec stimulus presentation
        # 1 sec delay
        # need to respond in .9 sec

        abba = np.random.random_sample() < .5
        pictureNumber = 20
        picToMatch = int(np.random.random_sample() * pictureNumber) 
        fillerElts = [i for i in range(pictureNumber) if i != picToMatch]
        seqLength  = 6 + int(np.random.random_sample() * (pictureNumber - 6))

        if abba:
            maxSeqLength = 30
            # repeats of the filler elements is allowed
            seq = [picToMatch] + random.choices(fillerElts, k=seqLength) + [picToMatch]
        else:
            seq = [picToMatch] + random.sample(fillerElts, k=seqLength) + [picToMatch]

        # TODO: find images, import them, put them into the array

        self.currFrame = 0
        return self.experiment[0]
