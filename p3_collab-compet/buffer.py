# -*- encoding: utf-8 -*-
from collections import deque
import random
import torch
import numpy as np

from config import DEVICE


class ReplayBuffer:
    def __init__(self, buffer_size=10000):
        self.buffer_size = buffer_size
        self.memory = deque(maxlen=buffer_size)

    def add(self, *args):
        self.memory.append(args)

    def sample(self, batch_size):
        samples = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)

        states = torch.from_numpy(np.array(states)).float().to(DEVICE)
        actions = torch.from_numpy(np.array(actions)).float().to(DEVICE)
        rewards = torch.from_numpy(np.array(rewards)).float().to(DEVICE)
        next_states = torch.from_numpy(np.array(next_states)).float().to(DEVICE)
        dones = torch.from_numpy(np.array(dones).astype(np.uint8)).float().to(DEVICE)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)
