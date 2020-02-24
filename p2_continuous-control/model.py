# -*- encoding: utf-8 -*-
"""
@File           :   model.py
@Time           :   2020_01_26-15:34:36
@Author         :   zhenwang
@Description    :
  - Version 1.0.0: File created.
"""
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def layer_init(layer, w_scale=1.0):
    # nn.init.orthogonal_(layer.weight.data)
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    layer.weight.data.uniform_(-lim, lim)
    layer.weight.data.mul_(w_scale)

    nn.init.constant_(layer.bias.data, 0)
    return layer


class Actor(nn.Module):
    def __init__(self, state_size, action_size,
                 seed=0,
                 hidden_units=(400, 300)):
        """
        Initialize parameters and build the actor model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_units (tuple): Dimensions of sequence hidden layers
        """
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size

        self.seed = torch.manual_seed(seed)

        if not hidden_units or not len(hidden_units):
            raise Exception(f"hidden_units({hidden_units}) should NOT be empty!")

        hidden_gate_func = nn.LeakyReLU

        layers = []
        previous_features = state_size
        for idx, hidden_size in enumerate(hidden_units):
            layers.append(layer_init(nn.Linear(previous_features, hidden_size)))
            # layers.append(nn.BatchNorm1d(hidden_size))  # adding batch norm
            layers.append(hidden_gate_func(inplace=True))

            previous_features = hidden_size

        layers.append(layer_init(nn.Linear(previous_features, action_size), 3e-3))
        layers.append(nn.Tanh())
        self.fc_body = nn.Sequential(*layers)

    def forward(self, state):
        return self.fc_body(state)


class Critic(nn.Module):
    def __init__(self, state_size, action_size,
                 seed=0,
                 hidden_units=(400, 300)):
        """
        Initialize parameters and build the critic model.
        Params
        =======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_units (tuple): Dimensions of sequence hidden layers
        """
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size

        self.seed = torch.manual_seed(seed)

        if not hidden_units or not len(hidden_units):
            raise Exception(f"hidden_units({hidden_units}) should NOT be empty!")

        hidden_gate_func = nn.LeakyReLU

        self.fc_body = nn.Sequential(
            nn.Linear(state_size, hidden_units[0]),
            hidden_gate_func(inplace=True),
        )

        layers = []
        previous_features = hidden_units[0] + action_size
        for hidden_size in hidden_units[1:]:
            layers.append(layer_init(nn.Linear(previous_features, hidden_size)))
            # layers.append(nn.BatchNorm1d(hidden_size))  # adding batch norm
            layers.append(hidden_gate_func(inplace=True))

            previous_features = hidden_size

        layers.append(layer_init(nn.Linear(previous_features, 1), 3e-3))
        # layers.append(nn.ReLU(inplace=True))  # using ReLU, because the value should NOT be negative.
        self.critic_body = nn.Sequential(*layers)

    def forward(self, state, action):
        x = self.fc_body(state)
        x = torch.cat((x, action), dim=1)
        return self.critic_body(x)


if __name__ == "__main__":
    import numpy as np
    import torch.optim as optim

    seed = 1
    n_episodes = 100

    np.random.seed(seed)

    actor_hidden_sizes = (400, 300)
    critic_hidden_sizes = (400, 300, 256)

    states_ = np.random.randn(20, 33)

    states = torch.from_numpy(states_).float()
    actor = Actor(33, 10, seed=seed, hidden_units=actor_hidden_sizes)
    actor_optimizer = optim.Adam(actor.parameters(), lr=1e-4)

    critic = Critic(33, 10, seed=seed, hidden_units=critic_hidden_sizes)
    critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3)
    q_target = torch.from_numpy(np.random.randn(20, 1) + 5.).float()

    for i_episode in range(1, 1 + n_episodes):
        states = torch.from_numpy(states_).float()

        actions = actor(states)

        actions_ = actions.detach().cpu().numpy()
        actions = torch.from_numpy(actions_).float()
        q_expected = critic(states, actions)
        '''
        torch.onnx.export(critic, (states, actions), 'critic_model.onnx', 
                          input_names=['states', 'actions','a','b','c','d','e','f'], 
                          output_names=['q_expected'])
        '''
        critic_loss = F.mse_loss(q_expected, q_target)

        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        q_expected = critic(states, actions)
        actor_loss = -q_expected.mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        print(f"episode {i_episode}: actor_loss={actor_loss}, critic_loss={critic_loss}")
    pass
