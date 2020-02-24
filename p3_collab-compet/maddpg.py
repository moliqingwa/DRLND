# -*- encoding: utf-8 -*-
from pathlib import Path
import numpy as np
import torch
from buffer import ReplayBuffer
from ddpg import DDPG
from config import DEVICE


class MADDPG(object):
    """
    The main class that defines and trains all the DDPG agents.
    """
    def __init__(self, num_agents, state_size, action_size,
                 buffer_size=int(1e6), batch_size=128, writer=None,
                 actor_hidden_sizes=(256, 128), actor_lr=1e-4, actor_weight_decay=0.,
                 critic_hidden_sizes=(256, 128), critic_lr=1e-3, critic_weight_decay=0.,
                 model_folder_path=None,
                 ):
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size

        self.full_state_size = num_agents * state_size
        self.full_action_size = num_agents * action_size

        # Replay memory
        self.memory = ReplayBuffer(buffer_size)

        # TensorboardX Writer
        self.writer = writer

        # Actor Network Parameters
        self.actor_hidden_sizes = actor_hidden_sizes
        self.actor_lr = actor_lr
        self.actor_weight_decay = actor_weight_decay

        # Critic Network Parameters
        self.critic_hidden_sizes = critic_hidden_sizes
        self.critic_lr = critic_lr
        self.critic_weight_decay = critic_weight_decay

        # Model Folder
        self.folder_path = Path() if model_folder_path is None else Path(model_folder_path)

        # MADDPG Agents
        self.agents = []
        self._init_agents()

    def reset(self):
        for agent in self.agents:
            agent.reset()

    def act(self, states, noise=0.):
        return [agent.act(obs, noise) for agent, obs in zip(self.agents, states)]

    def step(self, i_episode, states, actions, rewards, next_states, dones,
             tau=0.01, num_learns=1):

        # save to replay buffer
        self.memory.add(states, actions, rewards, next_states, dones)

        # train the model
        if len(self.memory) >= self.batch_size and num_learns > 0:
            actor_loss_list, critic_loss_list = [], []

            for _ in range(num_learns):  # learn multiple times at every step
                states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

                for agent_id in range(self.num_agents):
                    # Learn one time for the agents
                    actor_loss, critic_loss = self._learn(agent_id, states, actions, next_states, rewards, dones)

                    actor_loss_list.append(actor_loss)
                    critic_loss_list.append(critic_loss)

            # Record Losses for actor & critic
            if self.writer:
                for agent_id in range(self.num_agents):
                    self.writer.add_scalars(f'agent{agent_id}/losses',
                                            {'critic loss': np.mean(critic_loss_list),
                                             'actor_loss': np.mean(actor_loss_list)},
                                            i_episode)

            # Soft update
            self._update_all(tau)

    def save(self):
        for agent in self.agents:
            torch.save(agent.actor_local.state_dict(),
                       self.folder_path / f'checkpoint_actor_local_{agent.id}.pth')
            torch.save(agent.critic_local.state_dict(),
                       self.folder_path / f'checkpoint_critic_local_{agent.id}.pth')

    def load(self, agent_id=None):
        for agent in self.agents:
            agent_id_ = agent.id if agent_id is None else agent_id
            agent.actor_local.load_state_dict(torch.load(self.folder_path / f'checkpoint_actor_local_{agent_id_}.pth'))
            agent.critic_local.load_state_dict(torch.load(self.folder_path / f'checkpoint_critic_local_{agent_id_}.pth'))

    def _init_agents(self):
        for i in range(self.num_agents):
            agent = DDPG(i, self.state_size, self.full_state_size,
                         self.action_size, self.full_action_size,
                         self.actor_hidden_sizes, self.actor_lr, self.actor_weight_decay,
                         self.critic_hidden_sizes, self.critic_lr, self.critic_weight_decay)
            self.agents.append(agent)

    def _learn(self, agent_id, states, actions, next_states, rewards, dones):

        critic_full_actions, critic_full_next_actions = [], []
        for agent in self.agents:
            # current actions
            actor_actions = agent.actor_local(states[:, agent.id, :])
            critic_full_actions.append(actor_actions)

            # next actions
            actor_next_actions = agent.actor_target.forward(next_states[:, agent.id, :])
            critic_full_next_actions.append(actor_next_actions)

        # learn for the agent
        current_agent = self.agents[agent_id]
        actor_loss, critic_loss = current_agent.learn(states, actions, rewards, next_states, dones,
                                                      critic_full_actions, critic_full_next_actions)
        return actor_loss, critic_loss

    def _update_all(self, tau):
        for agent in self.agents:
            agent.update(agent.actor_local, agent.actor_target, tau)
            agent.update(agent.critic_local, agent.critic_target, tau)

