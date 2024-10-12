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

# we added these
import copy
from tqdm import tqdm

Transition = collections.namedtuple("Transition", field_names=['state', 'action', 'reward', 'next_state', 'dones'])

class ReplayMemory():
    def __init__(self, memory_size, batch_size):
        # define init params
        # use collections.deque
        # BEGIN STUDENT SOLUTION
        self.memory = collections.deque(maxlen=memory_size)
        self.batch_size = batch_size
        # END STUDENT SOLUTION

    def sample(self):
        if len(self.memory) < self.batch_size:
            print('ERROR: Not enough samples in replay buffer')
            return random.sample(self.memory, len(self.memory))
        
        return random.sample(self.memory, self.batch_size)


    def append(self, transition):
        # append to the collections.deque
        # BEGIN STUDENT SOLUTION
        self.memory.append(transition)
        # END STUDENT SOLUTION



class DeepQNetwork(nn.Module):
    def __init__(self, state_size, action_size, lr_q_net=2e-4, gamma=0.99, epsilon=0.05, target_update=50, burn_in=10000, replay_buffer_size=50000, replay_buffer_batch_size=32, device='cpu'):
        super(DeepQNetwork, self).__init__()

        # define init params
        self.state_size = state_size
        self.action_size = action_size

        self.gamma = gamma
        self.epsilon = epsilon

        self.target_update = target_update

        self.burn_in = burn_in

        self.device = device

        hidden_layer_size = 256

        # q network
        q_net_init = lambda: nn.Sequential(
            nn.Linear(state_size, hidden_layer_size),
            nn.ReLU(),
            # BEGIN STUDENT SOLUTION
            nn.Linear(hidden_layer_size, self.action_size)
            # nn.ReLU()
            # END STUDENT SOLUTION
        )

        # initialize replay buffer, networks, optimizer, move networks to device
        # BEGIN STUDENT SOLUTION
        self.replay_buffer = ReplayMemory(replay_buffer_size, replay_buffer_batch_size)

        # initialize network and optimizer
        self.q_omega = q_net_init()
        self.q_target = copy.deepcopy(q_net_init()) 
        self.q_target.load_state_dict(self.q_omega.state_dict()) 
        self.q_target.eval()

        self.optimizer = optim.Adam(self.q_omega.parameters(), lr_q_net)
        # END STUDENT SOLUTION

    def burn_traj(self, env):
        state, _ = env.reset()
        for _ in range(self.burn_in):
            action = env.action_space.sample()

            next_state, reward, done, _, _ = env.step(action)

            self.replay_buffer.append(Transition(torch.from_numpy(state).float(), action, reward, next_state, done))
            
            if done:
                state, _ = env.reset()
            else:
                state = next_state


    def forward(self, state):
        return (self.q_omega(state), self.q_target(state))


    def get_action(self, state, stochastic):
        # if stochastic, sample using epsilon greedy, else get the argmax
        # BEGIN STUDENT SOLUTION
        with torch.no_grad():
            omega_val, target_val = self.forward(torch.from_numpy(state))

        if stochastic and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            return omega_val.argmax().item()

        # END STUDENT SOLUTION



    def train(self):
        # train the agent using the replay buffer
        # BEGIN STUDENT SOLUTION
        if len(self.replay_buffer.memory) < self.replay_buffer.batch_size:
            return
        
        trans = self.replay_buffer.sample()
        minibatch = Transition(*zip(*trans))

        state_batch = torch.tensor(np.array(minibatch.state))
        action_batch = torch.tensor(minibatch.action)
        reward_batch = torch.tensor(minibatch.reward)
        dones = torch.tensor(np.array(minibatch.dones), dtype=torch.float32, device=self.device)

        state_action_values = self.q_omega(state_batch).gather(1, action_batch.unsqueeze(1))
        next_state_values = self.q_target(torch.from_numpy(np.array(minibatch.next_state))).max(1).values
        
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma * (1-dones)) + reward_batch
        
        # Compute the loss
        loss_omega = F.mse_loss(state_action_values.squeeze(1), expected_state_action_values)

        self.optimizer.zero_grad()
        loss_omega.backward()

        self.optimizer.step()


    def run(self, env, max_steps, num_episodes, train, init_buffer):
        total_rewards = []

        # initialize replay buffer
        # run the agent through the environment num_episodes times for at most max steps
        # BEGIN STUDENT SOLUTION
        if train:
            self.q_omega.train()
        else:
            self.q_omega.eval()
        
        # Initialize the replay buffer with experience using a random policy
        if init_buffer:
            self.burn_traj(env)

        c = 0 # number of steps
        for e in range(num_episodes):
            state, _ = env.reset()
            total_reward = 0
            for t in range(max_steps):
                action = self.get_action(state,stochastic=True)
                next_state, reward, done, _, _ = env.step(action)

                if train:
                    self.replay_buffer.append(Transition(torch.from_numpy(state).float(), action, reward, next_state, done))

                total_reward += reward
                state = next_state

                if train:
                    self.train()

                # if not done:
                #     state = next_state

                c = c + 1
                
                if done:
                    break

                # Update target policy
                if train and c % self.target_update == 0:
                    # deepcopy
                    self.q_target.load_state_dict(self.q_omega.state_dict())
            
            print(f"Episode {e} - Reward {total_reward}")
            total_rewards.append(total_reward)

        # END STUDENT SOLUTION
        return(total_rewards)



def graph_agents(graph_name, agents, env, max_steps, num_episodes):
    print(f'Starting: {graph_name}')

    # graph the data mentioned in the homework pdf
    # BEGIN STUDENT SOLUTION

    graph_every = 100
    eval_episodes = 20

    # Define the evaluation checkpoints based on the number of episodes
    eval_checkpoints = np.arange(graph_every, num_episodes + 1, graph_every)

    # Initialize the rewards matrix to store evaluation results
    D = np.zeros((len(agents), len(eval_checkpoints)))

    # Function to evaluate the agent at a checkpoint
    def evaluate_agent(agent, env, max_steps, eval_episodes):
        return np.mean(agent.run(env, max_steps, eval_episodes, train=False, init_buffer=False))

    # Loop through each agent
    for i, agent in enumerate(agents):
        print(f'Evaluating agent {i+1}/{len(agents)}')

        # Loop through evaluation checkpoints
        for j, checkpoint in tqdm(enumerate(eval_checkpoints), total=len(eval_checkpoints)):
            print(f'Agent {i+1}, Checkpoint: Episode {checkpoint}/{num_episodes}')

            # Train the agent until the next checkpoint
            init_buffer = (j == 0)  # Initialize buffer only at the first training step
            agent.run(env, max_steps, checkpoint - (eval_checkpoints[j - 1] if j > 0 else 0), train=True, init_buffer=init_buffer)

            # Evaluate the agent after training
            D[i, j] = evaluate_agent(agent, env, max_steps, eval_episodes)

    # Compute statistics for graphing
    average_total_rewards = np.mean(D, axis=0)
    min_total_rewards = np.min(D, axis=0)
    max_total_rewards = np.max(D, axis=0)
    

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
    env_name = args.env_name
    num_runs =args.num_runs
    num_episodes = args.num_episodes
    max_steps = args.max_steps
    # init args, agents, and call graph_agent on the initialized agents
    # BEGIN STUDENT SOLUTION
    env = gym.make(env_name)

    # run_total_rewards = np.zeros((num_runs,int(num_episodes/100)),dtype=object)

    # for run in range(num_runs):
    #     agent = DeepQNetwork(env.observation_space.shape[0], env.action_space.n)
    #     rewards = agent.run(env, max_steps, num_episodes, train=True, init_buffer=True)
    #     run_total_rewards[run] = rewards

    # graph_agents("I dunno", 0, 0, max_steps= max_steps, num_episodes = num_episodes, total_rewards=run_total_rewards)

    # Initialize agent
    agents = [DeepQNetwork(env.observation_space.shape[0],
                           env.action_space.n) for _ in range(args.num_runs)]

    # Train each agent
    for i, agent in enumerate(agents):
    #     print(f'Starting run {i+1}/{args.num_runs}')
        agent.run(env, max_steps=args.max_steps, num_episodes=args.num_episodes, train=True, init_buffer=True)

    # Graph the performance
    graph_agents(f'DQN_n{args.num_runs}', agents, env, args.max_steps, args.num_episodes)

    # END STUDENT SOLUTION



if '__main__' == __name__:
    main()
