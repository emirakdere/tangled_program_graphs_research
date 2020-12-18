import numpy as np
import gym
import matplotlib.pyplot as plt
import cv2
import random
import tpg

env = gym.make('gym_task:delayGo-v0')
# env = gym.make('gym_task:simplePerceptualDM-v0')
# env = gym.make('gym_task:contextDependentDM-v0')
# env = gym.make('gym_task:multisensoryDM-v0')
# env = gym.make('gym_task:parametricWorkingMem-v0')
# # env = gym.make('gym_task:delayedMatchToSample-v0') # on hold until everything else is working
# env = gym.make('gym_task:delayedMatchToCategory-v0')


print("Action Space:", env.action_space)

env.reset()
env.render()

def show_state(env):
	cv2.imshow('env', env.render())
	cv2.waitKey(5)


def getStateAs1DVector(state):
	return np.ndarray.flatten(state)

# import to do training
from tpg.trainer import Trainer
# import to run an agent (always needed)
from tpg.agent import Agent


# Source: as adapted from https://github.com/Ryan-Amaral/PyTPG/blob/master/tpg_examples.ipynb
import time # for tracking time
tStart = time.time()

# first create an instance of the TpgTrainer
# this creates the whole population and everything
# teamPopSize should realistically be at-least 100
trainer = Trainer(actions=range(env.action_space.n), teamPopSize=100)

curScores = [] # hold scores in a generation
summaryScores = [] # record score summaries for each gen (min, max, avg)

# 5 generations isn't much (not even close), but some improvements
# should be seen.
for gen in range(20): # generation loop
    curScores = [] # new list per gen
    
    agents = trainer.getAgents()
    
    while True: # loop to go through agents
        teamNum = len(agents)
        agent = agents.pop()
        if agent is None:
            break # no more agents, so proceed to next gen
        
        state = env.reset() # get initial state and prep environment
        score = 0
        for i in range(500): # run episodes that last 500 frames

            # show_state(env)
            
            # get action from agent
            act = agent.act(getStateAs1DVector(np.array(state, dtype=np.int32)))

            # feedback from env
            state, reward, isDone, debug = env.step(act)
            score += reward # accumulate reward in score
            if isDone:
                break # end early if losing state

        agent.reward(score) # must reward agent (if didn't already score)
            
        curScores.append(score) # store score
        
        if len(agents) == 0:
            break
            
    # at end of generation, make summary of scores
    summaryScores.append((min(curScores), max(curScores),
                    sum(curScores)/len(curScores))) # min, max, avg
    trainer.evolve()
    

print('Time Taken (Hours): ' + str((time.time() - tStart)/3600))
print('Results:\nMin, Max, Avg')
for result in summaryScores:
    print(result[0],result[1],result[2])
plt.plot([summaryScores[i][2] for i in range(len(summaryScores))])
plt.show()