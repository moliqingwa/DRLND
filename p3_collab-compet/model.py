# -*- encoding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _layer_init(layer, w_scale=1.0):
    # nn.init.orthogonal_(layer.weight.data)
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    layer.weight.data.uniform_(-lim, lim)
    layer.weight.data.mul_(w_scale)

    nn.init.constant_(layer.bias.data, 0)
    return layer


class Actor(nn.Module):
    def __init__(self, in_size, hidden_units, out_size, out_gate=nn.Tanh):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size

        hidden_gate_func = nn.ELU

        layers = []
        previous_features = in_size
        for idx, hidden_size in enumerate(hidden_units):
            layers.append(_layer_init(nn.Linear(previous_features, hidden_size)))
            # layers.append(nn.BatchNorm1d(hidden_size))  # adding batch norm
            layers.append(hidden_gate_func(inplace=True))

            previous_features = hidden_size

        layers.append(_layer_init(nn.Linear(previous_features, out_size), 3e-3))
        if out_gate is not None:
            layers.append(out_gate())
        self.fc_body = nn.Sequential(*layers)

    def forward(self, states):
        return self.fc_body(states)


class Critic(nn.Module):
    def __init__(self, in_size, full_action_size, hidden_units=(400, 300)):
        super().__init__()

        hidden_gate_func = nn.ELU

        self.fc_body = nn.Sequential(
            nn.Linear(in_size, hidden_units[0]),
            hidden_gate_func(inplace=True),
        )

        layers = []
        previous_features = hidden_units[0] + full_action_size
        for hidden_size in hidden_units[1:]:
            layers.append(_layer_init(nn.Linear(previous_features, hidden_size)))
            # layers.append(nn.BatchNorm1d(hidden_size))  # adding batch norm
            layers.append(hidden_gate_func(inplace=True))

            previous_features = hidden_size

        layers.append(_layer_init(nn.Linear(previous_features, 1), 3e-3))
        self.critic_body = nn.Sequential(*layers)

    def forward(self, full_states, full_actions):
        x = self.fc_body(full_states)
        x = torch.cat((x, full_actions), dim=1)
        return self.critic_body(x)


if __name__ == "__main__":
    import torch.nn.functional as F
    full_states = torch.from_numpy(np.random.rand(48).reshape(-1, 48)).float()
    actor_states = torch.from_numpy(np.random.rand(24).reshape(-1, 24)).float()
    full_actions = torch.from_numpy(np.random.rand(4).reshape(-1, 4)).float()
    target_value = torch.tensor(10.).view(-1, 1).float()

    actor = Actor(24, hidden_units=(256, 256), out_size=2)
    critic = Critic(48, 4, hidden_units=(256, 256))
    optimizer = torch.optim.Adam(critic.parameters())

    '''
    torch.onnx.export(actor, (actor_states, ), "actor.onnx", verbose=False,
                      training=False,
                      input_names=['actor_state', 'a', 'b', 'c', 'd', 'e', 'f'],
                      output_names=['action'])

    torch.onnx.export(critic, (full_states, full_actions), "critic.onnx", verbose=False,
                      training=False,
                      input_names=['full_states', 'full_actions', 'a', 'b', 'c', 'd', 'e', 'f'],
                      output_names=['q'])
    '''

    steps = 100
    for i_step in range(steps):
        predict_value = critic(full_states, full_actions)
        loss = F.mse_loss(predict_value, target_value)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Step: {i_step},\tLoss: {loss},\tPredict: {predict_value}")
