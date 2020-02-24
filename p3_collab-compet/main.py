# -- coding: utf-8 --
from collections import deque

import random
import numpy as np
from tensorboardX import SummaryWriter
from unityagents import UnityEnvironment
import matplotlib.pyplot as plt

from config import *
from maddpg import MADDPG


def seeding(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# maddpg training
def maddpg_training(maddpg, writer,
                    total_episodes=N_EPISODES, t_max=1000,
                    train_every_steps=TRAIN_EVERY_STEPS, num_learn_steps_per_step=NUM_LEARN_STEPS_PER_ENV_STEP,
                    eps_start=EPS_START, eps_decay=EPS_DECAY, eps_end=EPS_END,
                    noise=NOISE_START):
    scores_list = []
    avg_scores_list = []
    scores_deque_ = deque(maxlen=100)
    eps = eps_start
    for i_episode in range(total_episodes):
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        states = env_info.vector_observations  # get the current state (for each agent)

        maddpg.reset()

        scores = np.zeros(num_agents)  # initialize the score (for each agent)
        for i_step in range(t_max):
            if random.random() < eps:
                actions = [np.random.uniform(-1, 1, 2) for _ in range(num_agents)]
            elif i_episode > EPISODES_BEFORE_TRAINING:
                actions = maddpg.act(states, noise)  # get actions
                noise = max(noise * NOISE_REDUCTION, NOISE_END)
            eps = max(eps - eps_decay, eps_end)

            env_info = env.step(actions)[brain_name]  # send all actions to tne environment
            rewards = env_info.rewards  # get reward (for each agent)
            next_states = env_info.vector_observations  # get next state (for each agent)
            dones = env_info.local_done  # see if episode finished
            scores += rewards  # update the score (for each agent)

            # train the model
            if i_step % train_every_steps == 0:
                maddpg.step(i_episode,
                            states, actions, rewards, next_states, dones,
                            tau=TAU, num_learns=num_learn_steps_per_step)

            # exit loop if episode finished
            if all(dones):
                break

            states = next_states  # roll over states to next time step
            i_step += 1

        avg_score_over_100_episodes_ = np.mean(scores_deque_, 0) if scores_deque_ else [0 for _ in range(num_agents)]
        # record rewards
        for i in range(num_agents):
            writer.add_scalars(f'agent{i}/rewards',
                               {
                                   "episode_rewards": scores[i],
                                   "avg_rewards": avg_score_over_100_episodes_[i],
                               },
                               i_episode)

        scores_deque_.append(scores)
        scores_list.append(np.max(scores))
        avg_scores_list.append(max(avg_score_over_100_episodes_))
        print(
            f"Episode: {i_episode},EPS: {eps:.3f}, \tNoise: {noise:.3f},\tScores: [{scores[0]:.2f};{scores[1]:.2f}],\tAvg Scores: {max(avg_score_over_100_episodes_):.6f}")

        if max(avg_score_over_100_episodes_) > 1.3:
            break
    maddpg.save()

    return scores_list, avg_scores_list


if __name__ == "__main__":
    seeding(seed=SEED)

    env = UnityEnvironment(file_name="Tennis.app", no_graphics=True)
    brain_name = env.brain_names[0]
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
    print('The state for the first agent looks like:', states[0])

    full_action_size = num_agents * action_size
    full_state_size = num_agents * state_size

    writer = SummaryWriter(log_dir="logs/train", flush_secs=30)
    maddpg = MADDPG(num_agents, state_size, action_size,
                    buffer_size=BUFFER_SIZE, writer=writer)

    scores_list, avg_scores_list = maddpg_training(maddpg, writer)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores_list) + 1), scores_list)
    plt.plot(np.arange(1, len(avg_scores_list) + 1), avg_scores_list)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
    plt.savefig("Scores.png")

    writer.close()
    env.close()
