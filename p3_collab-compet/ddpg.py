# -*- encoding: utf-8 -*-
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from model import Actor, Critic
from ou_noise import OUNoise
from config import DEVICE

_critic_local = None
_critic_target = None


def get_critic(full_state_size, full_action_size, critic_hidden_sizes):
    global _critic_local, _critic_target
    if _critic_local is None or _critic_target is None:
        _critic_local = Critic(full_state_size, full_action_size, critic_hidden_sizes)
        _critic_target = Critic(full_state_size, full_action_size, critic_hidden_sizes)
    return _critic_local, _critic_target


class DDPG(object):
    """
    Interacts with and learns from the environment.

    There are two agents and the observations of each agent has 24 dimensions, while each agent's action has 2 dimensions.
    Here we use two separate actor networks (one for each agent using each agent's observations only and output that agent's action).
    The critic for each agents gets to see the full observations and full actions of all agents.
    """
    def __init__(self, agent_id,
                 state_size, full_state_size, action_size, full_action_size,
                 actor_hidden_sizes=(256, 128), actor_lr=1e-4, actor_weight_decay=0.,
                 critic_hidden_sizes=(256, 128), critic_lr=1e-3, critic_weight_decay=0.,
                 is_action_continuous=True):
        """
        Initialize an Agent object.

        :param agent_id (int): ID of each each agent.
        :param state_size (int): Dimension of each state for each agent.
        :param full_state_size (int): Dimension of full state for all agents.
        :param action_size (int): Dimension of each action for each agent.
        :param full_action_size: Dimension of full action for all agents.
        :param actor_hidden_sizes (tuple): Hidden units of the actor network.
        :param actor_lr (float): Learning rate of the actor network.
        :param actor_weight_decay (float): weight decay (L2 penalty) of the actor network.
        :param critic_hidden_sizes (tuple): Hidden units of the critic network.
        :param critic_lr (float): Learning rate of the critic network.
        :param critic_weight_decay (float): weight decay (L2 penalty) of the critic network.
        :param is_action_continuous (bool): Whether action space is continuous or discrete.
        """
        self.id = agent_id
        self.state_size = state_size
        self.full_state_size = full_state_size
        self.action_size = action_size
        self.full_action_size = full_action_size
        self.is_action_continuous = is_action_continuous

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, actor_hidden_sizes, action_size,
                                 out_gate=nn.Tanh if is_action_continuous else None)
        self.actor_target = Actor(state_size, actor_hidden_sizes, action_size,
                                  out_gate=nn.Tanh if is_action_continuous else None)
        self.update(self.actor_local, self.actor_target, 1.)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=actor_lr, weight_decay=actor_weight_decay)

        # Critic Network (w/ Target Network)
        num_agents = int(full_action_size/action_size)
        self.critic_local = Critic(full_state_size,
                                   full_action_size if is_action_continuous else num_agents,
                                   critic_hidden_sizes)
        self.critic_target = Critic(full_state_size,
                                    full_action_size if is_action_continuous else num_agents,
                                    critic_hidden_sizes)
        # self.critic_local, self.critic_target = get_critic(full_state_size, full_action_size, critic_hidden_sizes)
        self.update(self.critic_local, self.critic_target, 1.)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=critic_lr,
                                           weight_decay=critic_weight_decay)

        self.use_actor = True

        # Noise Process
        self.noise_scale = 0.
        self.noise = OUNoise(action_size)

    def reset(self):
        self.noise.reset()

    def act(self, state, noise_scale=0.0):
        """
        Returns action for given state using current policy.
        """
        states = torch.from_numpy(state[np.newaxis]).float()

        # calculate actions
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(states)
        self.actor_local.train()
        actions = actions.cpu().numpy().squeeze()

        # add noise
        actions += noise_scale * self.noise.sample()

        return np.clip(actions, -1, 1) if self.is_action_continuous else np.argmax(actions)

    def learn(self, states, actions, rewards, next_states, dones,
              full_actions_predicted, critic_full_next_actions,
              gamma=0.99):
        """
        Update policy and value parameters.
        Q_targets = r + γ * critic_target(next_state, action_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        :param states: Full states for training which size is (BATCHES, NUM_AGENTS, STATE_SIZE)
        :param actions: Full actions for training which size is (BATCHES, NUM_AGENTS, ACTION_SIZE)
        :param rewards: Full rewards for training which size is (BATCHES, NUM_AGENTS)
        :param next_states: Full next states for training which size is (BATCHES, NUM_AGENTS, STATE_SIZE)
        :param dones: Full dones for training which size is (BATCHES, NUM_AGENTS)
        :param full_actions_predicted:
        :param critic_full_next_actions: Full next states which size is (BATCHES, NUM_AGENTS * STATE_SIZE)
        :param gamma: discount ratio
        """
        full_states = states.view(-1, self.full_state_size)
        full_actions = actions.view(states.shape[0], -1).float()
        full_next_states = next_states.view(-1, self.full_state_size)
        critic_full_next_actions = torch.cat(critic_full_next_actions, dim=1).float().to(DEVICE)

        actor_rewards = rewards[:, self.id].view(-1, 1)
        actor_dones = dones[:, self.id].view(-1, 1)

        # ---------------------------- update critic ---------------------------- #
        q_next = self.critic_target.forward(full_next_states, critic_full_next_actions)

        q_target = actor_rewards + gamma * q_next * (1 - actor_dones)

        q_expected = self.critic_local(full_states, full_actions)

        # Compute critic loss
        critic_loss = F.mse_loss(q_expected, q_target.detach())

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        if self.use_actor:
            # detach actions from other agents
            full_actions_predicted = [actions if i == self.id else actions.detach()
                                      for i, actions in enumerate(full_actions_predicted)]
            full_actions_predicted = torch.cat(full_actions_predicted, dim=1).float().to(DEVICE)

            # Compute actor loss
            actor_loss = -self.critic_local.forward(full_states, full_actions_predicted).mean()

            # Minimize the loss
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
        else:
            actor_loss = torch.tensor(0)

        return actor_loss.cpu().item(), critic_loss.cpu().item()

    def update(self, source, target, tau=0.01):
        """
        Update target model parameters:
        θ_target = τ*θ_local + (1 - τ)*θ_target

        :param source: Pytorch model which parameters are copied from
        :param target: Pytorch model which parameters are copied to
        :param tau: interpolation parameter
        """
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(target_param.data * (1 - tau) + param.data * tau)

