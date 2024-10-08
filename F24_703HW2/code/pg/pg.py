#! python3

import argparse

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np # NOTE only imported because https://github.com/pytorch/pytorch/issues/13918
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical



class PolicyGradient(nn.Module):
    def __init__(self, state_size, action_size, lr_actor=1e-3, lr_critic=1e-3, mode='REINFORCE', n=128, gamma=0.99, device='cpu'):
        super(PolicyGradient, self).__init__()

        self.state_size = state_size
        self.action_size = action_size

        self.mode = mode
        self.n = n
        self.gamma = gamma

        self.device = device

        hidden_layer_size = 256

        # actor
        self.actor = nn.Sequential(
            nn.Linear(state_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, action_size),
            # BEGIN STUDENT SOLUTION
            nn.Softmax()
            # END STUDENT SOLUTION
        )

        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_size, hidden_layer_size),
            nn.ReLU(),
            # BEGIN STUDENT SOLUTION
            nn.Linear(hidden_layer_size, action_size),
            # END STUDENT SOLUTION
        )

        # initialize networks, optimizers, move networks to device
        # BEGIN STUDENT SOLUTION
        self.optim_actor = optim.Adam(self.actor.parameters(), lr = lr_actor)
        self.optim_critic = optim.Adam(self.critic.parameters(), lr = lr_critic)
        # END STUDENT SOLUTION


    def forward(self, state):
        return(self.actor(state), self.critic(state))


    def get_action(self, state, stochastic):
        # if stochastic, sample using the action probabilities, else get the argmax
        # BEGIN STUDENT SOLUTION
        actor_prob, critic_prob = self.forward(torch.from_numpy(state).float()) 
        cat = Categorical(actor_prob)
        if stochastic:
            # sample using action probabilities
            action = cat.sample()
            return action.item(), cat.log_prob(action)
        else:
            # sample using argmax
            action = torch.from_numpy(np.array(np.argmax(cat)))
            return action.item()

        # END STUDENT SOLUTION
        pass


    def calculate_n_step_bootstrap(self, rewards_tensor, values):
        # calculate n step bootstrap
        # BEGIN STUDENT SOLUTION
        # END STUDENT SOLUTION
        pass


    def train(self, states, actions, rewards, logprobs):
        # train the agent using states, actions, and rewards

        # BEGIN STUDENT SOLUTION
        # Vectorize
        T = len(rewards)
        G = np.zeros(T)
        for t in range(T):
            gammas = np.ones((T-1)-t) * self.gamma
            cumprod_gammas = np.cumprod(gammas) / self.gamma
            G[t] = np.sum(cumprod_gammas * rewards[t:-1])
        policy_loss = []
        for t in range(T):
            loss = -logprobs[t] * G[t] / T
            policy_loss.append(loss.unsqueeze(0))
        # L_theta = -(1/T) * np.sum(G * logprobs)
        # L_theta = torch.from_numpy(np.array(L_theta))
        # L_theta.requires_grad_()
        policy_loss = torch.cat(policy_loss).sum()
        #print("Policy Loss: ", policy_loss)

        self.optim_actor.zero_grad()
        policy_loss.backward()
        self.optim_actor.step()
        # END STUDENT SOLUTION


    def run(self, env, max_steps, num_episodes, train):
        total_rewards = []

        # run the agent through the environment num_episodes times for at most max steps
        # BEGIN STUDENT SOLUTION
        for e in range(num_episodes):
            # generate episode with max steps
            #print("[TRAIN] Training")
            states, actions, rewards, logprobs = self.generate_trajectory(env, max_steps, True)
            # train
            self.train(states, actions, rewards, logprobs)

            if e % 100 == 0:
                cumulative_reward = 0
                for _ in range(20):
                    # run test 
                    #print("[EVAL] Testing")
                    _, _, test_reward, _ = self.generate_trajectory(env, max_steps, False)
                    cumulative_reward += np.sum(test_reward)
                total_rewards.append(cumulative_reward / 20)
                print(cumulative_reward/20)
        # END STUDENT SOLUTION
        return np.array(total_rewards)

    # Self Made Function
    def generate_trajectory(self, env, max_steps, train):
        curr_state = env.reset()

        done = False
        steps = 0
        curr_state = env.unwrapped.state

        states   = []
        actions  = []
        rewards  = []
        logprobs = []

        while not done and steps < max_steps:
            action = None
            logprob = None
            if train:
                # then use stochastic
                action, logprob = self.get_action(curr_state, True)
                logprobs.append(logprob)
            else:
                action = self.get_action(curr_state, False)
            states.append(curr_state)
            actions.append(action)
            next_state, reward, done, _, _ = env.step(action)

            rewards.append(reward)
            curr_state = next_state
            steps = steps + 1
        #print("Trajectory: ", steps)
        return states, actions, rewards, logprobs
            
            
def graph_agents(graph_name, agents, env, max_steps, num_episodes, total_rewards):
    print(f'Starting: {graph_name}')

    # graph the data mentioned in the homework pdf
    # BEGIN STUDENT SOLUTION
    average_total_rewards = np.average(total_rewards, axis=1)
    graph_every = int(num_episodes / 100)
    min_total_rewards = np.min(total_rewards, axis=1)
    max_total_rewards = np.max(total_rewards, axis=1)

    print(average_total_rewards)
    print(min_total_rewards)
    print(max_total_rewards)
    # END STUDENT SOLUTION

    # plot the total rewards
    xs = [i * graph_every for i in range(len(average_total_rewards))]
    fig, ax = plt.subplots()
    plt.fill_between(xs, min_total_rewards, max_total_rewards, alpha=0.1)
    ax.plot(xs, average_total_rewards)
    ax.set_ylim(-max_steps * 0.01, max_steps * 1.1)
    ax.set_title(graph_name, fontsize=10)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Total Reward')
    fig.savefig(f'./graphs/{graph_name}.png')
    plt.close(fig)
    print(f'Finished: {graph_name}')



def parse_args():
    mode_choices = ['REINFORCE', 'REINFORCE_WITH_BASELINE', 'A2C']

    parser = argparse.ArgumentParser(description='Train an agent.')
    parser.add_argument('--mode', type=str, default='REINFORCE', choices=mode_choices, help='Mode to run the agent in')
    parser.add_argument('--n', type=int, default=64, help='The n to use for n step A2C')
    parser.add_argument('--num_runs', type=int, default=5, help='Number of runs to average over for graph')
    parser.add_argument('--num_episodes', type=int, default=3500, help='Number of episodes to train for')
    parser.add_argument('--max_steps', type=int, default=200, help='Maximum number of steps in the environment')
    parser.add_argument('--env_name', type=str, default='CartPole-v1', help='Environment name')
    return parser.parse_args()


def main():
    args = parse_args()
    max_steps = args.max_steps
    num_episodes = args.num_episodes
    mode = args.mode
    n = args.n
    num_runs = args.num_runs

    # init args, agents, and call graph_agents on the initialized agents
    # BEGIN STUDENT SOLUTION
    
    # Create Env
    env = gym.make(args.env_name, max_episode_steps=max_steps)
    policy_gradient = PolicyGradient(env.observation_space.shape[0], env.action_space.n , mode= mode,n= n)

    run_total_rewards = np.zeros((num_runs,5),dtype=object)
    print(run_total_rewards)
    for run in range(num_runs):
        run_rewards = policy_gradient.run(env, max_steps, num_episodes,True)
        print(run_rewards)
        print(run_rewards.shape)
        run_total_rewards[run] = run_rewards
    
    graph_agents("I dunno", 0, 0, max_steps= max_steps, num_episodes = num_episodes, total_rewards=run_total_rewards)
    # END STUDENT SOLUTION



if '__main__' == __name__:
    main()
