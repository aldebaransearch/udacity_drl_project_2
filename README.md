[//]: # (Image References)

[image1]: https://github.com/aldebaransearch/udacity_drl_project_2/blob/main/reacher.gif "Trained Agent"

# Project 2: Continuous Control

### Introduction

In this project, we train a deep reinforcement learning agent to navigate in the Unity environment Reacher. The report describing the solution in greater detail can be found in [report_2.pdf](https://github.com/aldebaransearch/udacity_drl_project_2/blob/main/report_2.pdf)  

![Trained Agent][image1]

The goal of the agent is to keep the end of an arm within a moving target region. For every time step where the agent succeeds a reward of +0.1 is provided. There are 2 versions of the environment; one with only one agent and another with 20 agents moving at the same time. The video above shows the 20 agent version using an altered version of the DDPG algorithm.  

### State and Action Space
The state space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. The action space contains four continuous variables ranging from -1 to 1:

### Solution
The agent is considered successful in the current setting, when it reaches an average score of 30 over 100 episodes. For the 20 agent case it is the average across the 20 agents, averaged across 100 time steps, that has to reach 30 for the environment to be considered solved. However, instead of stopping the agent at a score of 30, we run them a bit further to judge different training characteristics of the 2 environments.

### Getting Started
**`1`** Build a conda environment using the environment.yml file in this repository by running "conda env create -f environment.yml"

**`2`** Download the Unity environment from the link that matches your system:
 - **_Version 1: For the environment consisting of 1 agent_**
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

- **_Version 2: For the environment consisting of 20 agents_**
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

. Place the file in the project folder, and unzip it.

**`3`** Download the notebooks Continuous_Control_1.ipynb and Continuous_Control_20.ipynb as well as the python file util.py.

### Training
Simply follow the instructions in the first half of the Continuous_Control_1.ipynb/Continuous_Control_20.ipynb notebook. Rewards are saved in a file in a folder `results` (if that does not exist, you should create it) and checkpoints with neural network weights are saved at every 100th episode in the folder `checkpoints` (if that folder does not exist, you should create it).

### Check Solutions
To assess results from training, follow the instructions in the last half of the Continuous_Control_1.ipynb/Continuous_Control_20.ipynb notebook. 

If you want to examine precalculated solutions, download the `results` and `checkpoints` folders including all their contents and use them instead of your own training results.

### Important note for Linux users
Due to what seems to be a bug in the Unity environment running on Linux machines, the notebook kernel has to be restarted between different trainings or different model assessments.


