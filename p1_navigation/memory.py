# -*- coding: utf-8 -*-#
# ----------------------------------------------------------------------
# Name:         memory
# Description:  
# Author:       zhenwang
# Date:         2019/12/18
# ----------------------------------------------------------------------
import random
from collections import deque


class ReplayBuffer(object):
    def __init__(self, buffer_size=100000, seed=0):
        self._deque = deque(maxlen=buffer_size)

        self.seed = random.seed(seed)

    def add(self, value):
        self._deque.append(value)

    def sample(self, batch_size):
        sampled_batch = random.sample(self._deque, batch_size)
        return sampled_batch

    def __len__(self):
        return len(self._deque)
