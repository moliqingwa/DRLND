# -*- coding: utf-8 -*-#
# ----------------------------------------------------------------------
# Name:         dqn_agent
# Description:  
# Author:       zhenwang
# Date:         2019/12/15
# ----------------------------------------------------------------------
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from model import DQNNetwork
from priority_memory import PrioritizedReplayBuffer

seed_ = 0
np.random.seed(seed_)
torch.manual_seed(seed_)
random.seed(seed_)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent(object):
    def __init__(self, state_size, action_size,
                 batch_size=32,
                 learning_rate=1e-4,
                 gamma=0.9,
                 local_tau=2e-3,
                 training_steps=3,
                 replay_buffer_size=100000,
                 seed=0):
        """
        DQN agent constructor.

        :param state_size: dimension of each state
        :param action_size: dimension of each action
        :param batch_size: batch size
        :param learning_rate: learning rate
        :param gamma: parameter for setting the discounted experiences of future rewards
        :param local_tau: interpolation parameter for updating target model parameters (τ*θ_local + (1 - τ)*θ_target)
        :param training_steps: specifies whether or not to train model after these steps
        :param replay_buffer_size: size of the replay memory buffer
        """
        self.state_size = state_size
        self.action_size = action_size

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.local_tau = local_tau
        self.training_steps = training_steps
        self.replay_buffer_size = replay_buffer_size

        self._model_local = DQNNetwork(self.state_size, self.action_size).to(device)
        self._model_target = DQNNetwork(self.state_size, self.action_size).to(device)

        self._optimizer = optim.Adam(self._model_local.parameters(), lr=self.learning_rate)

        # self._memory = ReplayBuffer(buffer_size=replay_buffer_size, seed=seed_)
        self._memory = PrioritizedReplayBuffer(buffer_size=replay_buffer_size,
                                               alpha=0.6,
                                               beta=0.4,
                                               beta_increment_per_sampling=0.001,
                                               epsilon=0.01,
                                               abs_err_upper=1.,
                                               min_prob_lower=1.e-6,
                                               seed=seed)

        self._step = 0

    def step(self, state, action, reward, next_state, done):
        """
        Agent step update.

        :param state: current state
        :param action: current action
        :param reward: current reward after taking current action
        :param next_state: next state after taking current action
        :param done: species the current episode is finished or not
        """
        self._memory.add((state, action, reward, next_state, 1 if done else 0))

        self._step += 1
        should_training = self._step % self.training_steps == 0 and len(self._memory) > self.batch_size
        if should_training:
            return True, self._train()
        return False, 0

    def _train(self):
        # Step 1: fetch random mini-batch from replay memory
        tree_idx, experience, weights = self._memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = [torch.from_numpy(np.vstack(v)).float()
                                                        for v in zip(*experience)]

        # Step 2: calculate q experiences
        q_expected = self._model_local(states).gather(1, actions.long())

        # Vanilla DQN
        # q_target_values = self._model_target(next_states).detach().max(1)[0].unsqueeze(1)

        # Double DQN
        _q_target_max_actions = self._model_local(next_states).argmax(dim=1, keepdim=True)
        q_target_values = self._model_target(next_states).gather(1, _q_target_max_actions).detach()

        q_targets = rewards + self.gamma * q_target_values * (1 - dones)

        # loss = F.mse_loss(q_expected, q_targets)
        weights_ = torch.from_numpy(weights).float()
        # weights_ /= torch.sum(weights_)
        loss = F.mse_loss(weights_ * q_expected, weights_ * q_targets)

        # Step 3: batch update priority
        abs_errors = torch.sum(torch.abs(q_expected - q_targets), dim=1).cpu().detach()
        self._memory.batch_update(tree_idx, abs_errors.numpy())

        # Step 4: train
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        # Step 5: update target network
        self._soft_update(self._model_local, self._model_target, self.local_tau)

        return loss

    @staticmethod
    def _soft_update(model_local, model_target, local_tau):
        """
        Soft update target model parameters.

        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, local_param in zip(model_target.parameters(), model_local.parameters()):
            target_param.data.copy_(local_tau * local_param.data + (1.0 - local_tau) * target_param.data)

    def act(self, state, eps=None):
        """
        Returns actions for given state as specified policy.

        :param state: current state
        :param eps: epsilon, for epsilon-greedy action selection. if eps is None, use predicted action.
        :return: Selected action.
        """
        if not eps or np.random.uniform() > eps:
            states = torch.from_numpy(state).float().unsqueeze(0)

            self._model_local.eval()
            predicted_action_values = self._model_local(states).detach().cpu().numpy().squeeze()
            self._model_local.train()

            action = np.argmax(predicted_action_values)
        else:
            action = np.random.randint(self.action_size)
        return action

    def save(self, path, **kwargs):
        """
        Save model state and parametes into a file.

        :param path: specifies saved file path
        :param kwargs: specifies model arguments dict
        """
        torch.save(
            {
                "kwargs": kwargs,
                "local_model_state_dict": self._model_local.state_dict(),
                "target_model_state_dict": self._model_target.state_dict(),
                "memory.beta": self._memory.beta,
            },
            path)

    def load(self, path):
        """
        Load model from a file.

        :param path: Specifies the file path to load from.
        :return: The model epochs if exists
        """
        checkpoint = torch.load(path)

        local_state_dict = checkpoint["local_model_state_dict"]
        if local_state_dict:
            self._model_local.load_state_dict(local_state_dict)

        target_state_dict = checkpoint["target_model_state_dict"]
        if target_state_dict:
            self._model_target.load_state_dict(target_state_dict)
            self._model_target.eval()

        self._memory.beta = checkpoint["memory.beta"]

        return checkpoint['kwargs'] or {}


def dqn(env, brain_name, agent, n_epochs,
        checkpoint_file=None,
        epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.005,
        smoothed_window=100):
    smoothed_scores, scores, losses = [], [], []

    # STEP 1: load checkpoint.pt if exist.
    checkpoint_loaded, saved_kwargs = False, {}
    accumulated_epochs = 0
    if checkpoint_file and os.path.exists(checkpoint_file):
        saved_kwargs = agent.load(checkpoint_file)
        accumulated_epochs = saved_kwargs['epochs']
        epsilon = saved_kwargs['epsilon']
        scores = saved_kwargs['scores']
        smoothed_scores = saved_kwargs['smoothed_scores']
        checkpoint_loaded = True

    if not checkpoint_loaded:
        accumulated_epochs = 0
        epsilon = 1.

    # STEP 2: train the dqn model
    for i_epoch in range(n_epochs):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]

        step, score, loss = 0, 0, 0
        while True:
            action = agent.act(state, epsilon)

            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]  # get the reward
            done = env_info.local_done[0]  # see if episode has finished

            trained, loss_ = agent.step(state, action, reward, next_state, done)

            if trained:
                loss += loss_
            score += reward  # update the score
            state = next_state  # roll over the state to next time step

            step += 1
            if done:  # exit loop if episode finished
                break
        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        losses.append(loss)
        scores.append(score)
        smoothed_scores.append(np.mean(scores[-smoothed_window:]))

        print(f"epoch: {i_epoch}, eps: {epsilon:.3f}, steps: {step}, loss: {loss:.6f}, score: {np.mean(scores[-50:])}")

    # STEP 3: save checkpoint_file
    if checkpoint_file:
        agent.save(checkpoint_file,
                   epochs=accumulated_epochs + n_epochs,
                   epsilon=epsilon,
                   scores=scores,
                   smoothed_scores=smoothed_scores,
                   )

    return smoothed_scores, scores, losses


if __name__ == "__main__":
    from unityagents import UnityEnvironment
    import numpy as np

    env = UnityEnvironment(file_name="Banana.app", no_graphics=True)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents in the environment
    print('Number of agents:', len(env_info.agents))

    # number of actions
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)

    # examine the state space
    state = env_info.vector_observations[0]
    print('States look like:', state)
    state_size = len(state)
    print('States have length:', state_size)

    env_info = env.reset(train_mode=True)[brain_name]

    agent = Agent(state_size=state.shape[0],
                  action_size=brain.vector_action_space_size,
                  batch_size=32,
                  learning_rate=1e-4,
                  gamma=0.95,
                  replay_buffer_size=10000,
                  seed=seed_)

    smoothed_scores, scores, losses = dqn(env=env, brain_name=brain_name,
                                          agent=agent,
                                          n_epochs=1500,
                                          checkpoint_file="model.pt",
                                          epsilon=1.0,
                                          epsilon_decay=0.99,
                                          epsilon_min=0.005)

    import matplotlib.pyplot as plt

    fig = plt.figure(1)
    ax = fig.add_axes([0.1, 0.5, 0.8, 0.5])
    ax.plot(scores, 'b', label='scores')
    ax.plot(smoothed_scores, 'r', label='smoothed_scores')
    ax.set_xlabel("Episode")
    ax.set_ylabel("Score")
    ax.legend(loc='upper left', fontsize=8)
    plt.show()
