#! python3

import argparse
import collections
import random

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np # NOTE only imported because https://github.com/pytorch/pytorch/issues/13918
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class ReplayMemory():
    def __init__(self, memory_size, batch_size):
        # define init params
        # use collections.deque
        # BEGIN STUDENT SOLUTION

        # Initialize the replay memory as a deque with a max length of memory_size
        self.memory = collections.deque(maxlen=memory_size)

        # Store the batch size for sampling
        self.batch_size = batch_size

        # END STUDENT SOLUTION
        pass


    def sample_batch(self):
        # randomly chooses from the collections.deque
        # BEGIN STUDENT SOLUTION

        # Randomly sample a batch of transitions from the memory
        return random.sample(self.memory, self.batch_size)
        # END STUDENT SOLUTION
        pass


    def append(self, transition):
        # append to the collections.deque - FIFO approach
        # BEGIN STUDENT SOLUTION

        # Append new experience (transition) to the memory buffer
        self.memory.append(transition)

        # END STUDENT SOLUTION
        pass



class DeepQNetwork(nn.Module):
    def __init__(self, state_size, action_size, lr_q_net=2e-4, gamma=0.99, epsilon=0.05, target_update=50, burn_in=10000, replay_buffer_size=50000, replay_buffer_batch_size=32, device='cpu'):
        super(DeepQNetwork, self).__init__()

        # define init params
        self.state_size = state_size
        self.action_size = action_size

        self.gamma = gamma
        # The probability of taking a random action for exploration (ε-greedy exploration)
        self.epsilon = epsilon

        self.target_update = target_update

        #  Number of steps before the training begins, used to populate the replay memory
        self.burn_in = burn_in

        self.device = device

        hidden_layer_size = 256

        # q network
        q_net_init = lambda: nn.Sequential(
            nn.Linear(state_size, hidden_layer_size),
            nn.ReLU(),
            # BEGIN STUDENT SOLUTION
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, action_size),
            # TODO: include ReLU here??
            # nn.ReLU()
            # END STUDENT SOLUTION
        )

        # initialize replay buffer, networks, optimizer, move networks to device
        # BEGIN STUDENT SOLUTION

        # TODO: check if it can be initialized this way
        self.policy_network = q_net_init(state_size, action_size).to(device)
        self.target_network = q_net_init(state_size, action_size).to(device)
        self.target_network.load_state_dict(self.policy_network.state_dict())


        # TODO: verify the memory_size & batch_size
        memory = ReplayMemory(10000, 10000)

        '''
            Note on optimizer:
                * amsgrad: stochastic optimization method. helps with convergence.
                * TODO: verify if policy param can be passed in this way [state_size, action_size]
        '''
        optimizer = optim.Adam(self.policy_network.params(), lr=lr_q_net, amsgrad=True)
        
        global steps_done; steps_done = 0
        # END STUDENT SOLUTION


    def forward(self, state):
        return(self.q_net(state), self.target(state))


    def get_action(self, state, stochastic):
        # if stochastic, sample using epsilon greedy, else get the argmax
        # BEGIN STUDENT SOLUTION

        sample = random.random()
        if stochastic:
            pass 
        else: # epsilon-greedy
            # TODO: How to define the epsilon threshold
            # using initialized ep for now 
            steps_done += 1
            if sample > self.epsilon:
                '''
                    Note on torch.no_grad():
                        * disable gradient calculation
                        * TODO: understand more on this
                '''
                with torch.no_grad():
                    return self.policy_network(state).max(1).indices.view(1, 1)
            else:
                # return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
                # TODO: is this the way to do it?
                return torch.tensor([[env.action_space.sample()]])



        # END STUDENT SOLUTION
        pass


    def train(self):
        # train the agent using the replay buffer
        # BEGIN STUDENT SOLUTION
        # END STUDENT SOLUTION
        pass


    def run(self, env, max_steps, num_episodes, train, init_buffer):
        total_rewards = []

        # initialize replay buffer
        # run the agent through the environment num_episodes times for at most max steps
        # BEGIN STUDENT SOLUTION
        # END STUDENT SOLUTION
        return(total_rewards)



def graph_agents(graph_name, agents, env, max_steps, num_episodes):
    print(f'Starting: {graph_name}')

    # graph the data mentioned in the homework pdf
    # BEGIN STUDENT SOLUTION
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
    parser = argparse.ArgumentParser(description='Train an agent.')
    parser.add_argument('--num_runs', type=int, default=5, help='Number of runs to average over for graph')
    parser.add_argument('--num_episodes', type=int, default=1000, help='Number of episodes to train for')
    parser.add_argument('--max_steps', type=int, default=200, help='Maximum number of steps in the environment')
    parser.add_argument('--env_name', type=str, default='CartPole-v1', help='Environment name')
    return parser.parse_args()



def main():
    args = parse_args()

    # init args, agents, and call graph_agent on the initialized agents
    # BEGIN STUDENT SOLUTION
    '''
        TODO:
            * verify if we can make the environment global
    '''
    global env; env = gym.make("CartPole-v1")

    device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
    )
    
    n_actions = env.action_space.n
    state, info = env.reset()
    n_observations = len(state)
    
    # send in action_size & state_size
    DeepQNetwork(n_actions, n_observations)
    
    # END STUDENT SOLUTION



if '__main__' == __name__:
    main()
