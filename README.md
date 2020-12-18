# Tangled Program Graphs for Learning Cognitive Tasks

## Description

The project was partly inspired by the [Task representations in neural networks trained to perform many cognitive tasks](https://doi.org/10.1038/s41593-018-0310-2) paper, and partly by the [Emergent Tangled Graph Representations for Atari Game Playing Agents](http://link.springer.com/chapter/10.1007/978-3-319-55696-3_5) paper.

The project uses Tangled Program Graphs (TPG) to learn Cognitive Tasks that are usually taught animals such as monkeys and mice. The aim of the project is to see if this genetic algorithm can model the learning process of the animals. Once the model is modified to ensure reasonable success on single task learning, the model will be used to learn multiple tasks at once. This means that a single agent (a single TPG model) will have to produce common policies for different tasks.

## Usage

### Setting up a virtual environment:

virtualenv venv

### Activate environment:

source ./venv/bin/activate

### Install editable:
First, git clone both this repository and [PyTPG](https://github.com/Ryan-Amaral/PyTPG). 

After you setup and activate your virtual environment:


`pip install opencv-python-headless` # opencv-python-headless is used instead of the generic opencv library
`cd PyTPG` # cd to wherever you cloned this repository
`pip install -e .`
`cd ../tangled_program_graphs_research/gym-task` # cd to wherever you cloned this repository
`pip install -e .`
`cd ..`
`python3 train.py` # will make this command-line friendly, but for now you have to go into the file and change lines



## Work done and future trajectory

So far, we have implemented 8 tasks (one is pending on achieving success on the other ones) in OpenAI's [gym](https://gym.openai.com/docs/) framework. For details of the tasks, refer to the gym-tasks/gym_tasks/envs folder. Currently, the models are able to learn to not do any movements in the fixation period in every task, however they are not able to learn much further than that. There are a couple possible explanation for this:

### The models are unable to encode history

This is a possible explanation. I will be demoing simpler tasks where the action can be learned without any memory requirement to evaluate this explanation. Initially this seemed like a plausable reason, but after giving it more thought, I do not know if this is the main cause; TPGs' policies should be like state machines, and state machines should be well able to encode history. However, when one looks at the tasks that the model was trained on, they look like tasks that do not necessarily require memory. Even if this is not the current issue, this might be a future direction to consider to improve the TPG performance.

### Issues with the action space and reward function

Currently, I am using a discrete-action model, which means that the agent should be selecting an action from {0, 1, ..., 8}. In this model, the reward is binary, meaning the approximation of the goal are not appropriately rewarded, resulting in a flat gradient. I will be migrating my implementation to a continuous model so that a 25 degree mistake is not punished as severely as the 180 degree one. This should fix the learning problem.

### Rewards are sparse

This is closely related to the previous reason. I punish the model if it makes a movement before the end of the fixation period, and reward it if it waits where it should. However, the model rarely successfully executes the task: it learns to wait until the movement period, but then simply executes the incorrect action. 

After the main reason is investigated for the models failure to learn the tasks, they will be fixed, and the model will be trained on multiple tasks on once, and then analyzed as the [Natural Neuroscience](https://doi.org/10.1038/s41593-018-0310-2) did. 

