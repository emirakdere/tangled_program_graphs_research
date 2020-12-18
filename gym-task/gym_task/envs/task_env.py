import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from math import sin, cos, pi
from PIL import Image
import cv2
import random

import gym
from gym import error, spaces, utils
from gym.utils import seeding

CATEGORY = 1
CONTINUOUS = 2

class TaskEnv:

    def __init__(self):

        self.DIM_X = self.DIM_Y = 301
        self.TIME = 20
        self.experiment = np.zeros((self.TIME, self.DIM_Y, self.DIM_X))
        self.fixationTime = (0, self.TIME-3)
        self.stimulusTime = (2, 20)
        self.cue = None
        self.midX, self.midY = self.DIM_X//2, self.DIM_Y//2
        self.currFrame = 0

        self.action_space = spaces.Discrete(9) # TODO: demo continous action space as well (enables rewarding approximate performances)
        ### REMEMBER THIS EXISTS IF YOU WANT TO USE IT: 
        # self.observation_space = spaces.Box(-high, high, dtype=np.float32)


    def stimulusPos(self, radius, pieSizeInDegrees, direction=False):

        r = radius 
        randomNumber = np.random.random_sample()
        angleDegree = randomNumber * 360
        angleRadian = angleDegree // pieSizeInDegrees * (pi/(180/pieSizeInDegrees))

        if not direction:
            stimulusX = self.midX + int(cos(angleRadian) * r)
            stimulusY = self.midY - int(sin(angleRadian) * r)

            return stimulusX, stimulusY, randomNumber
        else:
            return cos(angleRadian), sin(angleRadian), randomNumber

    def animateExperiment(self):
        # Set up plotting
        fig = plt.figure()
        ax = plt.axes()  

        # Animation function
        def animate(i): 
            z = self.experiment[i]
            cont = plt.imshow(z, cmap='gray')
            return cont  

        anim = animation.FuncAnimation(fig, animate, frames=self.TIME, repeat=False)
        plt.show()
        
    def _putToMiddle(self, outputArray, imageName):
        
        middleArray = np.array(Image.open(imageName))
        middleArray = (middleArray.astype('float')//130.)
        if isinstance(middleArray[0, 0], np.ndarray) and len(middleArray[0, 0]) == 3:
            middleArray = middleArray[:, :, 0]

        middleArray = cv2.resize(middleArray, (self.DIM_X//3, self.DIM_Y//3))

        outputArray[self.DIM_X//3 +1:2*(self.DIM_X//3) +1, \
                  self.DIM_Y//3 +1:2*(self.DIM_Y//3) +1] = middleArray


    def _dotExperiment(self, cohesion, dotNumber, stimulusTime, dotColor=1., angle=-1, pieSizeInDegrees=2):

        dotsX, dotsY = self.DIM_X//3, self.DIM_Y//3

        area = dotsX * dotsY

        dots = np.random.choice(area, dotNumber, replace=False)
        dotsInSameDir = np.random.choice(dotNumber, \
                                        int(dotNumber * cohesion), \
                                        replace=False)

        dotsPos = {index:[pos // dotsY + self.DIM_X//2 - self.DIM_X//6, \
                         (pos % dotsY) + self.DIM_Y//2 - self.DIM_Y//6] \
                            for index, pos in enumerate(dots)}

        # predetermine a set direction cohesive dots will move in
        if angle == -1:
            sameDirX, sameDirY, randomNumber = self.stimulusPos(None, pieSizeInDegrees, direction=True)
        else:
            sameDirX, sameDirY, randomNumber = cos(angle), sin(angle), None



        for i in dotsPos:
            row, col = dotsPos[i]

            if i in dotsInSameDir:
                dirX, dirY = sameDirX, sameDirY
            else:
                dirX, dirY, _ = self.stimulusPos(None, 2, direction=True)

            dotsPos[i].append(dirX)
            dotsPos[i].append(dirY)

        for t in range(stimulusTime[0], stimulusTime[1]):
            for i in dotsPos:

                self.experiment[t, int(dotsPos[i][1]), int(dotsPos[i][0])] = 0

                dotsPos[i][0] = (dotsPos[i][0] + dotsPos[i][2]*3)
                dotsPos[i][1] = (dotsPos[i][1] + dotsPos[i][3]*3)

                dotsPos[i][0] -= self.DIM_X//3 if dotsPos[i][0] > 2*(self.DIM_X//3) else 0
                dotsPos[i][1] -= self.DIM_Y//3 if dotsPos[i][1] > 2*(self.DIM_Y//3) else 0

                dotsPos[i][0] += self.DIM_X//3 if dotsPos[i][0] < (self.DIM_X//3) else 0
                dotsPos[i][1] += self.DIM_Y//3 if dotsPos[i][1] < (self.DIM_Y//3) else 0

                self.experiment[t, int(dotsPos[i][1]), int(dotsPos[i][0])] = dotColor


        return randomNumber

    def _step(self, action, categoryOrContinuousDecision):

        self.currFrame += 1
        experimentOver = False

        # fixation time
        if self.currFrame < self.fixationTime[1]:
            # punishment for movement in fixation time
            if action <= 7:
                reward = -100
                experimentOver = True

            # reward for no movement in fixation time
            else:
                reward = 1

        # movement time
        else:

            if categoryOrContinuousDecision == CATEGORY:
                correctAction = (self.aim >= 4 and 8 > action >= 4) or (self.aim < 4 and action < 4)
            else:
                correctAction = self.aim == action

            # if no movement, small punishment
            if action == 8 and self.currFrame != self.TIME: 
                reward = -1

            # if movement to correct location, reward.
            elif correctAction:
                reward = 100
                experimentOver = True

            # if movement to the wrong location (or no movement till the end), punishment
            else:
                reward = -50
                experimentOver = True

        return self.experiment[self.currFrame-1], reward, experimentOver, {}


    def _categoryStep(self, action):
        return self._step(action, CATEGORY)

    def _continuousStep(self, action):
        return self._step(action, CONTINUOUS)

    def render(self, mode='human'):
        return self.experiment[self.currFrame]

    def close(self):
        # TODO?
        pass



