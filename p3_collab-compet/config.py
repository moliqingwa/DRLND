# -*- encoding: utf-8 -*-
import torch

SEED = 1

N_EPISODES = 2300

GAMMA = 0.99  # discount ratio
TAU = 0.01  # interpolation factor of local model and target model

EPS_START = 1.0
EPS_DECAY = 3e-4
EPS_END = 0.0005

# amplitude of OU noise
# this slowly decreases to NOISE_END, total_count = log(NOISE_END/NOISE_START)/log(NOISE_REDUCTION)
NOISE_START = 2
NOISE_REDUCTION = 0.9997  # 0.999 -> 2995, 0.9997 -> 9985, 0.9998 -> 14978, 0.9999 -> 29956
NOISE_END = 0.1

EPISODES_BEFORE_TRAINING = 100  #

TRAIN_EVERY_STEPS = 1
NUM_LEARN_STEPS_PER_ENV_STEP = 3  # how many times training of the step

ACTOR_HIDDEN_UNITS = (256, 256)
ACTOR_LR = 1e-4
ACTOR_WEIGHT_DECAY = 1e-5

CRITIC_HIDDEN_UNITS = (256, 256)
CRITIC_LR = 5e-4
CRITIC_WEIGHT_DECAY = 1e-5

BATCH_SIZE = 256
BUFFER_SIZE = int(1e6)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
