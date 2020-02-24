# -*- coding: utf-8 -*-#
# ----------------------------------------------------------------------
# Name:         model
# Description:  
# Author:       zhenwang
# Date:         2019/12/15
# ----------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.functional as F


class PixelDQNNetwork(nn.Module):
    def __init__(self, channal_size, action_size):
        super().__init__()

        self.channal_size = channal_size
        self.action_size = action_size

        self.hidden = nn.Sequential(
            nn.Conv2d(channal_size, 16, kernel_size=5, stride=2),  # n x c x 84 x 84 -> n x 16 x 40 x 40
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # n x 16 x 41 x 41 -> n x 16 x 20 x 20

            nn.Conv2d(16, 32, kernel_size=3, stride=1),  # n x 16 x 20 x 20 -> n x 32 x 18 x 18
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # n x 32 x 18 x 18 -> n x 32 x 9 x 9

            nn.Conv2d(32, 64, kernel_size=3, stride=1),  # n x 32 x 9 x 9 -> n x 64 x 7 x 7
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # n x 64 x 7 x 7 -> n x 64 x 3 x 3
        )

        self.fc1 = nn.Sequential(
            nn.Linear(64*3*3, 32),
            nn.LeakyReLU(),
        )

        self.V = nn.Linear(32, 1)
        self.A = nn.Linear(32, action_size)

    def forward(self, state):
        hidden = self.hidden(state)
        hidden = hidden.view((hidden.size(0), -1))  # n x 576 (= 64 x 3 x 3)

        out = self.fc1(hidden)
        value = self.V(out)
        advantage = self.A(out)

        return value.expand_as(advantage) + (advantage - advantage.mean(dim=1, keepdim=True))

