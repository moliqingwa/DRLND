# -- coding: utf-8 --
import random
import numpy as np
from unityagents import UnityEnvironment

from config import *
from maddpg import MADDPG


def seeding(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    seeding(seed=SEED)

    env = UnityEnvironment(file_name="Tennis.app", no_graphics=False)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=False)[brain_name]

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

    maddpg = MADDPG(num_agents, state_size, action_size, buffer_size=0)
    maddpg.load(agent_id=1)

    for i_episode in range(10):
        env_info = env.reset(train_mode=False)[brain_name]

        i_step = 0
        scores = np.zeros(num_agents)  # initialize the score (for each agent)
        while True:
            actions = maddpg.act(states)

            env_info = env.step(actions)[brain_name]  # send all actions to tne environment
            rewards = env_info.rewards  # get reward (for each agent)
            next_states = env_info.vector_observations  # get next state (for each agent)
            dones = env_info.local_done  # see if episode finished
            scores += rewards  # update the score (for each agent)
            # exit loop if episode finished
            if all(dones):
                break

            states = next_states  # roll over states to next time step
            i_step += 1
        print(f"Episode: {i_episode},\tSteps: {i_step},\tAgent_0 Score: {scores[0]:.2f},\tAgent_1 Score:{scores[1]:.2f}")

    env.close()
