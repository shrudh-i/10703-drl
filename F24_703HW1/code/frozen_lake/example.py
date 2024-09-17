#! python3

import random
import warnings

import gymnasium

import lake_info



class RandomAgent():
    def __init__(self, number_of_discrete_actions):
        self.number_of_discrete_actions = number_of_discrete_actions

    def get_action(self, observation):
        return(random.randrange(self.number_of_discrete_actions))



def run(agent, env, max_steps):
    observation, info = env.reset()
    episode_observations, episode_actions, episode_rewards = [], [], []

    for _ in range(max_steps):
        action = agent.get_action(observation)

        next_observation, reward, terminated, truncated, info = env.step(action)

        episode_observations.append(observation)
        episode_actions.append(action)
        episode_rewards.append(reward)

        observation = next_observation

        if terminated:
            break

    return(episode_observations, episode_actions, episode_rewards)



def main():
    # frozen lake documentation:
    # https://gymnasium.farama.org/environments/toy_text/frozen_lake/

    # frozen lake code:
    # https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/toy_text/frozen_lake.py

    # you can remove render_mode='human' to not display the environment
    max_steps=10

    env = gymnasium.make('FrozenLake-v1',
                         desc=None,
                         map_name='4x4',
                         is_slippery=False,
                         max_episode_steps=max_steps,
                         render_mode='human')
    action_space = env.action_space
    observation_space = env.observation_space

    print()
    print(f'action space: {action_space}')
    print(f'observation space: {observation_space}')

    print()
    print(f'height of map: {env.unwrapped.nrow}')
    print(f'width of map: {env.unwrapped.ncol}')

    print()
    print(f'env.unwrapped.P[state][action] = [(probability of transition,  next observation, reward, terminal), ...]')
    print(f'env.unwrapped.P[1][1] = {env.unwrapped.P[1][1]}')
    print('try turning is_slippery to True and seeing how it affects the above output')

    agent = RandomAgent(number_of_discrete_actions=action_space.n)

    # ignore pygame avx2 support warnings
    with warnings.catch_warnings():
        observations, actions, rewards = run(agent, env, max_steps)

    print()
    print('The state numbers start from 0 in the top left corner and increase left to right and then top to bottom.')
    print('So the top right corner is 3 and the state immediatley below the top left corner is 4.')

    print()
    for observation, action, reward in zip(observations, actions, rewards):
        print(f'Agent was at state {observation} and chose action {lake_info.actions_to_names[action].ljust(5)} getting reward {reward}')

    print()
    print(f'Cummulative total rewards: {sum(rewards)}')



if __name__ == '__main__':
    main()
