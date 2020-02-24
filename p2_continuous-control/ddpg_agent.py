# -*- encoding: utf-8 -*-
"""
@File           :   ddpg_agent.py
@Time           :   2020_01_26-20:08:22
@Author         :   zhenwang
@Description    :
  - Version 1.0.0: File created.
"""
import copy
import random
from collections import deque, namedtuple

import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from model import Actor, Critic
from memory import ReplyBuffer

from config import *


class Agent(object):
    """
    Interacts with and learns from the environment.
    """

    def __init__(self, state_size, action_size, num_agents,
                 seed=0, buffer_size=int(1e6),
                 actor_lr=1e-4, actor_hidden_sizes=(128, 256), actor_weight_decay=0,
                 critic_lr=1e-4, critic_hidden_sizes=(128, 256, 128), critic_weight_decay=0,
                 batch_size=128, gamma=0.99, tau=1e-3):
        """
        Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            num_agents (int): number of agents to train
            seed (int): random seed, default value is 0
            buffer_size (int): buffer size of experience memory, default value is 100000

            actor_lr (float): learning rate of actor model, default value is 1e-4
            actor_lr (float): learning rate of actor model, default value is 1e-4
            actor_hidden_sizes (tuple): size of hidden layer of actor model, default value is (128, 256)
            critic_lr (float): learning rate of critic model, default value is 1e-4
            critic_hidden_sizes (tuple): size of hidden layer of critic model, default value is (128, 256, 128)

            batch_size (int): mini-batch size
            gamma (float): discount factor
            tau (float): interpolation parameter
        """
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.seed = seed

        self.batch_size = batch_size  # mini-batch size
        self.gamma = gamma  # discount factor
        self.tau = tau  # for soft update of target parameters

        # Actor Network
        self.actor_local = Actor(state_size, action_size, seed,
                                 hidden_units=actor_hidden_sizes).to(DEVICE)
        self.actor_target = Actor(state_size, action_size, seed,
                                  hidden_units=actor_hidden_sizes).to(DEVICE)
        self.actor_target.eval()
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(),
                                          lr=actor_lr,
                                          weight_decay=actor_weight_decay)

        # Critic Network
        self.critic_local = Critic(state_size, action_size, seed,
                                   hidden_units=critic_hidden_sizes).to(DEVICE)
        self.critic_target = Critic(state_size, action_size, seed,
                                    hidden_units=critic_hidden_sizes).to(DEVICE)
        self.critic_target.eval()
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(),
                                           lr=critic_lr,
                                           weight_decay=critic_weight_decay)

        # Noise process
        self.noise = OUNoise((num_agents, action_size), seed)

        # Replay memory
        self.memory = ReplyBuffer(buffer_size=buffer_size, seed=seed)

        # copy parameters of the local model to the target model
        self.soft_update(self.critic_local, self.critic_target, 1.)
        self.soft_update(self.actor_local, self.actor_target, 1.)

        self.seed = random.seed(seed)
        np.random.seed(seed)

        self.reset()

    def reset(self):
        self.noise.reset()

    def act(self, state, add_noise=True):
        # actions = np.random.randn(self.num_agents, self.action_size)
        # actions = np.clip(actions, -1, 1)

        state = torch.from_numpy(state).float().to(DEVICE)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def step(self, states, actions, rewards, next_states, dones):
        """
        Save experience in replay memory, and use random sample from buffer to learn.
        """

        # Save experience / reward
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample(batch_size=self.batch_size)
            self.learn(experiences, self.gamma)

    def learn(self, experiences, gamma, last_action_loss=None):
        """
        Update policy and experiences parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-experiences

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ------- update critic ------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        q_targets = rewards + (gamma * q_targets_next * (1 - dones))
        q_targets = q_targets.detach()

        # Compute critic loss
        q_expected = self.critic_local(states, actions)
        assert q_expected.shape == q_targets.shape
        critic_loss = F.mse_loss(q_expected, q_targets)

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1.0)  # clip the gradient (Udacity)
        self.critic_optimizer.step()

        # ------- update actor ------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        #  update target networks
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)

        return actor_loss.item(), critic_loss.item()

    def soft_update(self, local_model, target_model, tau):
        """
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.detach_()
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def save(self):
        """
        Save model state
        """
        torch.save(self.actor_local.state_dict(), "checkpoints/checkpoint_actor.pth")
        torch.save(self.actor_target.state_dict(), "checkpoints/checkpoint_actor_target.pth")
        torch.save(self.critic_local.state_dict(), "checkpoints/checkpoint_critic.pth")
        torch.save(self.critic_target.state_dict(), "checkpoints/checkpoint_critic_target.pth")

    def load(self):
        """
        Load model state
        """
        self.actor_local.load_state_dict(torch.load("checkpoints/checkpoint_actor.pth", map_location=lambda storage, loc: storage))
        self.actor_target.load_state_dict(torch.load("checkpoints/checkpoint_actor_target.pth", map_location=lambda storage, loc: storage))
        self.critic_local.load_state_dict(torch.load("checkpoints/checkpoint_critic.pth", map_location=lambda storage, loc: storage))
        self.critic_target.load_state_dict(torch.load("checkpoints/checkpoint_critic_target.pth", map_location=lambda storage, loc: storage))

    def __str__(self):
        return f"{str(self.actor_local)}\n{str(self.critic_local)}"


class OUNoise:
    """
    Ornstein-Uhlenbeck process.
    """

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """
        Initialize parameters and noise process.
        """
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """
        Reset the internal state (= noise) to mean (mu).
        """
        self.state = copy.copy(self.mu)

    def sample(self):
        """
        Update internal state and return it as a noise sample.
        """
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state


def ddpg(n_episodes=100,
         eps_start=1.0, eps_decay=1e-5, eps_end=0.05,
         max_t=10000, learn_every_step=100):
    scores_deque = deque(maxlen=100)
    scores, actor_losses, critic_losses = [], [], []
    eps = eps_start
    for i_episode in range(1, 1 + n_episodes):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations

        agent.reset()

        avg_score = 0
        actor_loss_list, critic_loss_list = [], []
        for t in range(max_t):
            actions = agent.act(states, add_noise=random.random() < eps)
            eps = max(eps - eps_decay, eps_end)

            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done

            agent.step(states, actions, rewards, next_states, dones)

            # Learn, if enough samples are available in memory
            if len(agent.memory) > agent.batch_size and \
                    t % learn_every_step == 0:
                for _ in range(3):
                    experiences = agent.memory.sample(batch_size=agent.batch_size)
                    actor_loss, critic_loss = agent.learn(experiences, agent.gamma,
                                                          last_action_loss=actor_loss_list[-1] if actor_loss_list else None)
                    actor_loss_list.append(actor_loss)
                    critic_loss_list.append(critic_loss)

            avg_score += np.mean(rewards)
            states = next_states
            if np.any(dones):
                break

        scores_deque.append(avg_score)
        scores.append(avg_score)
        actor_losses.append(np.mean(actor_loss_list))
        critic_losses.append(np.mean(critic_loss_list))
        print(f"\rEpisode {i_episode}\tExploration: {eps:.6f}\t"
              f"Average Score: {np.mean(scores_deque):.2f}\tCurrent Score: {avg_score:.2f}\t"
              f"Actor Loss: {np.mean(actor_loss_list):.2e}\tCritic Loss: {np.mean(critic_loss_list):.2e}")

        if i_episode % 100 == 0:
            # agent.save()
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))

        if np.mean(scores_deque) > 30 and len(scores_deque) >= 100:
            agent.save()
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            break

    return scores, actor_losses, critic_losses


if __name__ == "__main__":
    import os
    import json
    import datetime
    from unityagents import UnityEnvironment
    import matplotlib.pyplot as plt
    from torchviz import make_dot

    env = UnityEnvironment(file_name=UNITY_APP_FILE_NAME,
                           seed=SEED,
                           no_graphics=UNITY_APP_NO_GRAPHICS)

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)

    # size of each action
    action_size = brain.vector_action_space_size

    # examine the state space
    states = env_info.vector_observations
    state_size = states.shape[1]

    agent = Agent(state_size=state_size,
                  action_size=action_size,
                  num_agents=num_agents,
                  seed=SEED, buffer_size=MEMORY_BUFFER_SIZE,
                  actor_lr=ACTOR_LR, actor_hidden_sizes=ACTOR_HIDDEN_UNITS, actor_weight_decay=ACTOR_WEIGHT_DECAY,
                  critic_lr=CRITIC_LR, critic_hidden_sizes=CRITIC_HIDDEN_UNITS, critic_weight_decay=CRITIC_WEIGHT_DECAY,
                  batch_size=BATCH_SIZE, gamma=GAMMA, tau=TAU)
    print(agent)

    agent.load()
    env_info = env.reset(train_mode=True)[brain_name]
    states = env_info.vector_observations
    '''  # watch the smart agent
    for _ in range(100):
        t_step, total_rewards = 0, 0
        while True:
            actions = agent.act(states, add_noise=False)

            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done

            states = next_states
            t_step += 1
            total_rewards += np.mean(rewards)
            if np.any(dones):
                break

        print(t_step, total_rewards)
    '''
    scores, actor_losses, critic_losses = ddpg(n_episodes=N_EPISODES,
                                               eps_start=EPS_START, eps_decay=EPS_DECAY, eps_end=EPS_END,
                                               max_t=MAX_STEPS, learn_every_step=LEARN_EVERY_STEP)

    agent.save()

    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    ax1.plot(np.arange(1, len(scores) + 1), scores)
    ax1.set_ylabel('Score')
    ax1.set_xlabel('Episode #')

    ax2 = fig.add_subplot(312)
    ax2.plot(np.arange(1, len(actor_losses) + 1), actor_losses)
    # ax2.legend()
    ax2.set_ylabel('Actor Loss')
    ax2.set_xlabel('Episode #')

    ax3 = fig.add_subplot(313)
    ax3.plot(np.arange(1, len(critic_losses) + 1), critic_losses)
    ax3.set_ylabel('Critic Loss')
    ax3.set_xlabel('Episode #')

    plt.show()

    from file_utils import get_vars_from_file
    now_time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.mkdir(now_time_str)
    plt.savefig(f"{now_time_str}/figure.png")
    param_dict = get_vars_from_file("config.py", raise_exception=False)
    with open(f"{now_time_str}/params.txt", 'w') as f:
        f.write(json.dumps(param_dict))

    env.close()
