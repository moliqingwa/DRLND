# -- coding: utf-8 --
from collections import deque, defaultdict
from pathlib import Path

import random
import numpy as np
from tensorboardX import SummaryWriter
from unityagents import UnityEnvironment
import matplotlib.pyplot as plt

from config import *
from buffer import ReplayBuffer
from ddpg import DDPG

# ============== CONFIGURATION ============== #
N_EPISODES = 1500

GAMMA = 0.99  # discount ratio
TAU = 0.01  # interpolation factor of local model and target model

EPS_START = 1.0
EPS_DECAY = 3e-4
EPS_END = 0.0005

# amplitude of OU noise
# this slowly decreases to NOISE_END, total_count = log(NOISE_END/NOISE_START)/log(NOISE_REDUCTION)
NOISE_START = 2
NOISE_REDUCTION = 0.9997  # 0.999 -> 2995, 0.9997 -> 9985, 0.9998 -> 14978, 0.9999 -> 29956
NOISE_END = 0.1

EPISODES_BEFORE_TRAINING = 10  #

TRAIN_EVERY_STEPS = 1
NUM_LEARN_STEPS_PER_ENV_STEP = 3  # how many times training of the step

ACTOR_HIDDEN_UNITS = (256, 256, 64)
ACTOR_LR = 1e-4
ACTOR_WEIGHT_DECAY = 1e-5

CRITIC_HIDDEN_UNITS = (256, 256, 64)
CRITIC_LR = 5e-4
CRITIC_WEIGHT_DECAY = 1e-5

BATCH_SIZE = 256
BUFFER_SIZE = int(2e6)


# ============== START HERE ============== #

def seeding(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# maddpg training
def maddpg_training(maddpg_dict, writer,
                    total_episodes=N_EPISODES, t_max=1000,
                    train_every_steps=TRAIN_EVERY_STEPS, num_learn_steps_per_step=NUM_LEARN_STEPS_PER_ENV_STEP,
                    eps_start=EPS_START, eps_decay=EPS_DECAY, eps_end=EPS_END,
                    noise=NOISE_START):
    scores_list_dict = {}
    avg_scores_list_dict = {}
    scores_deque_dict = {}
    for brain_name in maddpg_dict:
        scores_list_dict[brain_name] = []
        avg_scores_list_dict[brain_name] = []
        scores_deque_dict[brain_name] = deque(maxlen=100)

    eps = eps_start
    for i_episode in range(total_episodes):
        states_dict = {}
        for brain_name, maddpg in maddpg_dict.items():
            env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
            states = env_info.vector_observations  # get the current state (for each agent)
            states_dict[brain_name] = states

            maddpg.reset()

        scores = np.zeros((len(maddpg_dict.keys()), num_agents))  # initialize the score (for each agent)
        for i_step in range(t_max):
            actions_dict = {}
            for brain_name, maddpg in maddpg_dict.items():
                if random.random() < eps:
                    actions = [np.random.randint(action_size) for _ in range(num_agents)]
                else:
                    actions = maddpg.act(states_dict[brain_name], noise)  # get actions
                actions_dict[brain_name] = actions
            eps = max(eps - eps_decay, eps_end)
            if i_episode > EPISODES_BEFORE_TRAINING:
                noise = max(noise * NOISE_REDUCTION, NOISE_END)

            total_dones = []
            env_info_dict = env.step(actions_dict)  # send all actions to tne environment
            for i, (brain_name, maddpg) in enumerate(maddpg_dict.items()):
                env_info = env_info_dict[brain_name]
                rewards = env_info.rewards  # get reward (for each agent)
                next_states = env_info.vector_observations  # get next state (for each agent)
                dones = env_info.local_done  # see if episode finished

                scores[i] += rewards  # update the score (for each agent)
                total_dones += dones

                # train the model
                if i_step % train_every_steps == 0:
                    maddpg.step(i_episode,
                                states_dict[brain_name], actions_dict[brain_name], rewards, next_states, dones,
                                tau=TAU, num_learns=num_learn_steps_per_step)

                states_dict[brain_name] = next_states   # roll over states to next time step
            # exit loop if episode finished
            if all(total_dones):
                break

            i_step += 1

        for j, brain_name in enumerate(maddpg_dict):
            avg_score_over_100_episodes_ = np.mean(scores_deque_dict[brain_name], 0) if scores_deque_dict[brain_name] else [0 for _ in range(num_agents)]
            # record rewards
            for i in range(num_agents):
                writer.add_scalars(f'{brain_name}_{i}/rewards',
                                   {
                                       "episode_rewards": scores[j][i],
                                       "avg_rewards": avg_score_over_100_episodes_[i],
                                   },
                                   i_episode)

            scores_deque_dict[brain_name].append(scores[j])
            scores_list_dict[brain_name].append(np.max(scores[j]))
            avg_scores_list_dict[brain_name].append(max(avg_score_over_100_episodes_))
        print(
            f"Episode: {i_episode},EPS: {eps:.3f}, \tNoise: {noise:.3f},\tScores: [{max(scores[0]):.2f};{max(scores[1]):.2f}],\tAvg Scores: {max(avg_score_over_100_episodes_):.6f}")

    for _, maddpg in maddpg_dict.items():
        maddpg.save()

    return scores_list_dict, avg_scores_list_dict


class MADDPG(object):
    """
    The main class that defines and trains all the DDPG agents.
    """
    def __init__(self, brain_name, num_agents, state_size, action_size,
                 buffer_size=int(1e6), batch_size=128, writer=None,
                 actor_hidden_sizes=(256, 128), actor_lr=1e-4, actor_weight_decay=0.,
                 critic_hidden_sizes=(256, 128), critic_lr=1e-3, critic_weight_decay=0.,
                 model_folder_path=None,
                 ):
        self.brain_name = brain_name
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
                    self.writer.add_scalars(f'{self.brain_name}_{agent_id}/losses',
                                            {'critic loss': np.mean(critic_loss_list),
                                             'actor_loss': np.mean(actor_loss_list)},
                                            i_episode)

            # Soft update
            self._update_all(tau)

    def save(self):
        for agent in self.agents:
            torch.save(agent.actor_local.state_dict(),
                       self.folder_path / f'{self.brain_name}_actor_local_{agent.id}.pth')
            torch.save(agent.critic_local.state_dict(),
                       self.folder_path / f'{self.brain_name}_critic_local_{agent.id}.pth')

    def load(self, agent_id=None):
        for agent in self.agents:
            agent_id_ = agent.id if agent_id is None else agent_id
            agent.actor_local.load_state_dict(torch.load(self.folder_path / f'{self.brain_name}_actor_local_{agent.id}.pth'))
            agent.critic_local.load_state_dict(torch.load(self.folder_path / f'{self.brain_name}_critic_local_{agent_id_}.pth'))

    def _init_agents(self):
        for i in range(self.num_agents):
            agent = DDPG(i, self.state_size, self.full_state_size,
                         self.action_size, self.full_action_size,
                         self.actor_hidden_sizes, self.actor_lr, self.actor_weight_decay,
                         self.critic_hidden_sizes, self.critic_lr, self.critic_weight_decay,
                         is_action_continuous=False)
            self.agents.append(agent)

    def _learn(self, agent_id, states, actions, next_states, rewards, dones):

        critic_full_actions, critic_full_next_actions = [], []
        for agent in self.agents:
            # current actions
            actor_actions = agent.actor_local(states[:, agent.id, :]).argmax(dim=1, keepdim=True)
            critic_full_actions.append(actor_actions)

            # next actions
            actor_next_actions = agent.actor_target.forward(next_states[:, agent.id, :]).argmax(dim=1, keepdim=True)
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


if __name__ == "__main__":
    import os
    import time
    import shutil

    log_folder = "logs_soccer"
    if os.path.exists(log_folder):
        os.rename(log_folder, "{}.backup.{}".format(log_folder, time.strftime("%Y-%m-%d_ %H-%M-%S", time.localtime())))
        shutil.rmtree(log_folder, ignore_errors=True)

    seeding(seed=SEED)

    writer = SummaryWriter(log_dir=f"{log_folder}/train", flush_secs=30)
    env = UnityEnvironment(file_name="Soccer.app", no_graphics=True)
    maddpg_dict = {}
    for brain_name in env.brain_names:
        print("="*5, brain_name, "="*5)
        brain = env.brains[brain_name]

        # reset the environment
        env_info = env.reset(train_mode=True)[brain_name]

        # number of agents
        num_agents = len(env_info.agents)
        print('Number of agents:', num_agents)

        # size of each action
        action_size = brain.vector_action_space_size
        print('Size of each action:', action_size)

        # examine the state space
        states = env_info.vector_observations
        state_size = states.shape[1]
        print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
        print('The state for the first agent looks like:', states[0].shape)

        full_action_size = num_agents * action_size
        full_state_size = num_agents * state_size

        maddpg = MADDPG(brain_name, num_agents, state_size, action_size,
                        buffer_size=BUFFER_SIZE, writer=writer,
                        model_folder_path=f"{log_folder}")
        maddpg_dict[brain_name] = maddpg

    scores_list_dict, avg_scores_list_dict = maddpg_training(maddpg_dict, writer)

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # plt.plot(np.arange(1, len(scores_list) + 1), scores_list)
    # plt.plot(np.arange(1, len(avg_scores_list) + 1), avg_scores_list)
    # plt.ylabel('Score')
    # plt.xlabel('Episode #')
    # plt.show()
    # plt.savefig("Soccer_Scores.png")

    writer.close()
    env.close()
