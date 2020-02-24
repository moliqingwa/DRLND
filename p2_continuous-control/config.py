#!/Users/zhenwang/software/anaconda3/envs/ai/bin/python
# -*- encoding: utf-8 -*-
"""
@File           :   config.py
@Time           :   2020_01_28-17:28:33
@Author         :   zhenwang
@Description    :
  - Version 1.0.0: File created.
"""
import torch

SEED = 1

UNITY_APP_FILE_NAME = "Reacher20.app"
UNITY_APP_NO_GRAPHICS = False

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


ACTOR_LR = 1e-4  # learning rate
ACTOR_HIDDEN_UNITS = (256, 128)
ACTOR_WEIGHT_DECAY = 1e-5

CRITIC_LR = 1e-3  # learning rate
CRITIC_HIDDEN_UNITS = (256, 128)
CRITIC_WEIGHT_DECAY = 1e-5

MEMORY_BUFFER_SIZE = int(1e6)  # maximum size of replay buffer
BATCH_SIZE = 128  # mini-batch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # interpolation parameter

N_EPISODES = 100  # total episodes to train
EPS_START = 1.0  # initial value for exploration (epsilon)
EPS_DECAY = 2e-5  # epsilon decay value after each step
EPS_END = 0.05  # lower limit of epsilon
MAX_STEPS = 10000  # maximum training steps of each epoch
LEARN_EVERY_STEP = BATCH_SIZE  # extra learning after every step
