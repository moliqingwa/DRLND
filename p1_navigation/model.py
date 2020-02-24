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


class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size,
                 hidden_features=64,
                 dueling=True):
        super().__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.dueling = dueling

        if dueling:
            self.hidden = nn.Sequential(
                nn.Linear(state_size, hidden_features),
                nn.LeakyReLU(),

                nn.Linear(hidden_features, hidden_features),
                nn.LeakyReLU(),
            )
            self.V = nn.Linear(hidden_features, 1)
            self.A = nn.Linear(hidden_features, action_size)
        else:
            self.feature = nn.Sequential(
                nn.Linear(state_size, hidden_features),
                nn.ReLU(),

                nn.Linear(hidden_features, hidden_features),
                nn.ReLU(),

                nn.Linear(hidden_features, hidden_features),
                nn.ReLU(),

                nn.Linear(hidden_features, action_size),
            )

    def forward(self, state):
        if self.dueling:
            hidden = self.hidden(state)
            value = self.V(hidden)
            advantage = self.A(hidden)

            return value.expand_as(advantage) + (advantage - advantage.mean(dim=1, keepdim=True))
        else:
            return self.feature(state)
