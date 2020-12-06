import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from math import sin, cos, pi
from PIL import Image
import cv2
import random

class Tasks:
	def __init__(self):

		self.DIM_X = self.DIM_Y = 301
		self.TIME = 20
		self.experiment = np.zeros((self.TIME, self.DIM_Y, self.DIM_X))
		self.fixationTime = (0, self.TIME-3)
		self.stimulusTime = (2, 20)
		self.cue = None
		self.midX, self.midY = self.DIM_X//2, self.DIM_Y//2
		


	def stimulusPos(self, radius, pieSizeInDegrees, direction=False):

		r = radius 
		angleDegree = np.random.random_sample() * 360
		angleRadian = angleDegree // pieSizeInDegrees * (pi/(180/pieSizeInDegrees))

		if not direction:
			stimulusX = self.midX + int(cos(angleRadian) * r)
			stimulusY = self.midY - int(sin(angleRadian) * r)
			return stimulusX, stimulusY
		else:
			return cos(angleRadian), sin(angleRadian)

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
		# anim.save('simplePerceptualDM.mp4', writer=FFwriter, fps=15)
		plt.show()

	def delayGo(self): #1, memory guided response

		self.TIME = 10
		self.experiment = np.zeros((self.TIME, self.DIM_Y, self.DIM_X))
		self.fixationTime = (0, self.TIME-3)
		self.stimulusTime = (2, 4) 


		# add the fixation point
		self.experiment[self.fixationTime[0]:self.fixationTime[1], self.midY-1:self.midY+2, self.midX-1:self.midX+2] = .5
		
		stimulusX, stimulusY = self.stimulusPos(.6 * self.DIM_X / 2, 45)

		self.experiment[self.stimulusTime[0]:self.stimulusTime[1], stimulusY-1:stimulusY+2, stimulusX-1:stimulusX+2] = .1
		self.animateExperiment()


	def _dotExperiment(self, cohesion, dotNumber, stimulusTime, dotColor=1., angle=-1):

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
		sameDirX, sameDirY = self.stimulusPos(None, 2, direction=True) if angle == -1 else cos(angle), sin(angle)


		for i in dotsPos:
			row, col = dotsPos[i]

			if i in dotsInSameDir:
				dirX, dirY = sameDirX, sameDirY
			else:
				dirX, dirY = self.stimulusPos(None, 2, direction=True)

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

	def simplePerceptualDM(self):

		self.TIME = 30
		self.experiment = np.zeros((self.TIME, self.DIM_Y, self.DIM_X))
		# self.fixationTime = (0, self.TIME-3)
		stimulusTime = (2, self.TIME-5) 

		self._dotExperiment(.5, 50, stimulusTime)

		self.animateExperiment()


	def _putToMiddle(self, outputArray, imageName):
		
		middleArray = np.array(Image.open(imageName))
		middleArray = (middleArray.astype('float')//130.)
		if isinstance(middleArray[0, 0], np.ndarray) and len(middleArray[0, 0]) == 3:
			middleArray = middleArray[:, :, 0]

		middleArray = cv2.resize(middleArray, (self.DIM_X//3, self.DIM_Y//3))

		outputArray[self.DIM_X//3 +1:2*(self.DIM_X//3) +1, \
				  self.DIM_Y//3 +1:2*(self.DIM_Y//3) +1] = middleArray

	
	def contextDependentDM(self):
		# Fixation (0.5 s) Cue (1s) Stimulus (<3 s) Response
		halfSec = self.TIME//8
		self.fixationTime = (0, self.TIME - halfSec)
		self.cue = (halfSec, halfSec + 2*halfSec)
		self.stimulusTime = (self.cue[1], self.TIME - 2*halfSec)

		cueShapeInMiddle = np.zeros((self.DIM_Y, self.DIM_X))

		moveBasedOnDirection = np.random.random_sample() > .5
		if moveBasedOnDirection:
			self._putToMiddle(cueShapeInMiddle, "X.jpg")
		else:
			self._putToMiddle(cueShapeInMiddle, "O.jpg")

		self.experiment[self.cue[0]:self.cue[1]] = cueShapeInMiddle

		color = .33333333 + 2 * np.random.random() / 3 
		self._dotExperiment(1., 50, self.stimulusTime, dotColor=color)
		self.experiment[self.fixationTime[0]:self.fixationTime[1], self.midY-1:self.midY+2, self.midX-1:self.midX+2] = 1.

		self.animateExperiment()

		stimulusX, stimulusY = self.stimulusPos(.6 * self.DIM_X / 2, 45)


	# def _frequency(self, rate, interval):

	# 	halfSec = self.TIME // 8
	# 	flashDuration = halfSec * 4 // (rate * 2 - 1)
	# 	for i in range(0, rate*2-1, 2):
	# 		self.experiment[halfSec+i*flashDuration:halfSec+(i+1)*flashDuration, \
	# 						self.DIM_X//3:self.DIM_X//3*2, \
	# 						self.DIM_Y//3:self.DIM_Y//3*2] = 1.


	def multisensoryDM(self):
		rateThreshold = 3
		rate = 20
		self.stimulusTime = (halfSec, self.TIME - halfSec)

		halfSec = self.TIME // 8
		flashDuration = halfSec * 4 // (rate * 2 - 1)
		for i in range(0, rate*2-1, 2):
			self.experiment[halfSec+i*flashDuration:halfSec+(i+1)*flashDuration, \
							self.DIM_X//3:self.DIM_X//3*2, \
							self.DIM_Y//3:self.DIM_Y//3*2] = 1.

		self.animateExperiment()


	def parametricWorkingMem(self):

		rate1 = 15
		rate2 = 10

		halfSec = self.TIME // 8
		flashDuration1 = halfSec * 2 // (rate1 * 2 - 1)
		flashDuration2 = halfSec * 2 // (rate2 * 2 - 1)
		for i in range(0, rate1*2-1, 2):
			# print(i+1)
			self.experiment[halfSec+i*flashDuration1:halfSec+(i+1)*flashDuration1, \
							self.DIM_X//3:self.DIM_X//3*2, \
							self.DIM_Y//3:self.DIM_Y//3*2] = 1.

		for i in range(0, rate2*2-1, 2):
			# print(i+1)
			self.experiment[halfSec*5+i*flashDuration2:halfSec*5+(i+1)*flashDuration2, \
							self.DIM_X//3:self.DIM_X//3*2, \
							self.DIM_Y//3:self.DIM_Y//3*2] = 1.
		self.animateExperiment()

	def inhibitoryControl(self):
		#i.e., antisaccade
		halfSec = self.TIME//8

		fixationTime = (0, 5*halfSec)
		stimulusTime = (2*halfSec, self.TIME)
		colorCue     = (5*halfSec, self.TIME)

		antisaccade = np.random.random_sample() < .5
		targetLoc = [self.midY, 5*(self.DIM_X//6)]

		self.experiment[fixationTime[0]:fixationTime[1], self.midY-1:self.midY+2, self.midX-1:self.midX+2] = 1.
		self.experiment[stimulusTime[0]:stimulusTime[1], targetLoc[0]-3:targetLoc[0]+4, targetLoc[1]-3:targetLoc[1]+4] = 1.
		self.experiment[colorCue[0]:colorCue[1], self.midY-1:self.midY+2, self.midX-1:self.midX+2] = .25 if antisaccade else .75

		self.animateExperiment()

	def delayedMatchToSample(self):
		# .5 sec stimulus presentation
		# 1 sec delay
		# need to respond in .9 sec

		abba = np.random.random_sample() < .5
		pictureNumber = 20
		picToMatch = int(np.random.random_sample() * pictureNumber) 
		fillerElts = [i for i in range(pictureNumber) if i != picToMatch]
		seqLength  = 6 + int(np.random.random_sample() * (pictureNumber - 6))

		if abba:
			# print('abba')
			maxSeqLength = 30
			seq = [picToMatch] + random.choices(fillerElts, k=seqLength) + [picToMatch]
		else:
			seq = [picToMatch] + random.sample(fillerElts, k=seqLength) + [picToMatch]

		# print(seq)

		# find images, import them, put them into the array

		return

	def delayedMatchToCategory(self):
		halfSecond = self.TIME // 8
		stimulusTime = [halfSecond, 3*halfSecond]
		secondStimulusTime = [5*halfSecond, 7*halfSecond]

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

		print(categoryOf1 == categoryOf2)

		self.animateExperiment()













taskClass = Tasks()
# taskClass.delayGo() #1
# taskClass.simplePerceptualDM() #2
# taskClass.contextDependentDM() #3
# taskClass.multisensoryDM() #4
# taskClass.parametricWorkingMem() #5
# taskClass.inhibitoryControl() #6
# taskClass.delayedMatchToSample() # 7
taskClass.delayedMatchToCategory() # 8

