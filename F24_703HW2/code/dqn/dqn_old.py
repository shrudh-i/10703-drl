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

#experience replay
Transition = collections.namedtuple("Transition", ("state", "action", "next_state", "reward"))

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


    def sample_batch(self):
        # randomly chooses from the collections.deque
        # BEGIN STUDENT SOLUTION

        # Randomly sample a batch of transitions from the memory
        return random.sample(self.memory, self.batch_size)
    
        # END STUDENT SOLUTION


    def append(self, transition):
        # append to the collections.deque - FIFO approach
        # BEGIN STUDENT SOLUTION

        # Append new experience (transition) to the memory buffer
        self.memory.append(transition)

        # END STUDENT SOLUTION



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
            nn.Linear(hidden_layer_size, action_size)
            # END STUDENT SOLUTION
        )

        # initialize replay buffer, networks, optimizer, move networks to device
        # BEGIN STUDENT SOLUTION

        self.batch_size = 35
        self.memory_size = 50000

        self.q_net = q_net_init().to(self.device)
        self.q_target = q_net_init().to(self.device) # clone of q_net

        # copy the parameters for the q_net to q_target
        self.q_target.load_state_dict(self.q_net.state_dict())
        self.q_target.eval()

        self.memory = ReplayMemory(self.memory_size, self.batch_size)

        self.loss = nn.MSELoss()

        # amsgrad: stochastic optimization method. helps with convergence
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr_q_net)#, amsgrad=True)
        
        self.c = 0 # number of steps

        self.perform_burn_in()

        # END STUDENT SOLUTION

    def perform_burn_in(self):
            """
            Fill the replay memory with random actions by running the environment.
            """
            print(f"Starting burn-in for {self.burn_in} steps.")
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

            for _ in range(self.burn_in):
                # Select a random action
                action = torch.tensor([[env.action_space.sample()]]).to(self.device)

                # Take the action in the environment
                next_state, reward, done, _, _ = env.step(action.item())

                # Convert everything to tensors
                next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(self.device)
                reward = torch.tensor([reward], dtype=torch.float32).to(self.device)
                

                # Store transition in replay buffer
                self.memory.append(Transition(state, action, next_state, reward))

                # Reset the environment if episode ends
                if done:
                    state, _ = env.reset()
                    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                else:
                    state = next_state

                # print("Burn-in complete.")


    def forward(self, state):
        return(self.q_net(state), self.target(state))


    def get_action(self, state, stochastic):
        # if stochastic, sample using epsilon greedy, else get the argmax
        # BEGIN STUDENT SOLUTION

        sample = random.random()
        # state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        state = state.clone().detach().unsqueeze(0).to(self.device)

        if stochastic: # greedy policy (use episilon-greedy)
            if sample > self.epsilon:
                # Greedy approach
                ''' Note on torch.no_grad():
                        * disable gradient calculation
                '''
                with torch.no_grad():
                    # print("Greedy")
                    action = torch.argmax(self.q_net(state)).view(1, 1)
                    #print("ep greedy: ", action.item())
                    #print(self.q_net(state))
                    return torch.tensor([[action.item()]])
            else:
                # Explore
                # print("Explore")
                # return torch.tensor([[env.action_space.sample()]], device = self.device, dtype=torch.long)
                # print("ep greedy else: ", env.action_space.sample())
                # return torch.tensor([[env.action_space.sample()]], device = self.device, dtype=torch.long)
                return torch.tensor([[env.action_space.sample()]])

        else: # deterministic policy (just the argmax)
            with torch.no_grad():
                    action = torch.argmax(self.q_target(state)).view(1, 1)
                    # print("greedy: ", action.item())
                    # return action.item()
                    return torch.tensor([[action.item()]])
                
        # END STUDENT SOLUTION


    def train(self):
        # train the agent using the replay buffer
        # BEGIN STUDENT SOLUTION
        if len(self.memory.memory) < self.batch_size:
            return 0
        
        sample_transitions = self.memory.sample_batch()
        # print("Sample",sample_transitions)
        batch = Transition(*zip(*sample_transitions))
        # print(batch)
        # otherwise case: there is a next_state available
        minibatch = torch.tensor(list(map(lambda s: s is not None, batch.next_state)), device = self.device, dtype = torch.bool)
        #minibatch_next = torch.cat([s for s in batch.next_state if s is not None])
        # print("batch_state",batch.state)
        # print("batch_action", batch.action)
        state_batch = torch.cat(batch.state)
        # print(batch.action)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        predicted_values = self.q_net(state_batch).gather(1, action_batch)

        minibatch_next_values = torch.zeros(self.batch_size, device=self.device)
        # get the argmax (otherwise case for yi)
        minibatch_next_values[-1] = torch.max(self.q_target(state_batch[-1])).view(1,1)
        
        #Q(s, a) = reward(s, a) + Q(s_t+1, a_t+1)* gamma
        target_values = (minibatch_next_values * self.gamma) + reward_batch

        loss = self.loss(predicted_values, target_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()

        for param in self.q_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss
        # END STUDENT SOLUTION


    # def run(self, env, max_steps, num_episodes, train, init_buffer):
    def run(self, env, max_steps, num_episodes):
        total_rewards = []

        # initialize replay buffer
        # run the agent through the environment num_episodes times for at most max steps
        # BEGIN STUDENT SOLUTION

        self.c = 0 # number of steps

        for e in range(num_episodes): 
            rewards = 0
            state, _ = env.reset()
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

            for ms in range(max_steps):

                action = self.get_action(state, stochastic=True)
                # print("action",action)
                # if type(action) == torch.Tensor():
                #     print("ACTION :",action)
                # print("ACTION",action.item())
                next_state, reward, done, log, _ = env.step(action.item())
                # next_state, reward, done, _ = torch.FloatTensor([next_state]).to(self.device), torch.FloatTensor([reward]).to(self.device), done, log
                next_state, reward, done, _ = torch.tensor([next_state], dtype=torch.float32).to(self.device), torch.tensor([reward], dtype=torch.float32).to(self.device), done, log
                # print(next_state, reward, done)
                
                # if done:
                #     next_state = None
                
                self.memory.append(Transition(state, action, next_state, reward))
                state = next_state

                # TODO: finish self.train
                # handles lines 10 to 14 of Algorithm 4 in DQN 
                self.train()
                
                # TODO: how does this work?
                rewards += reward.detach().item()

                if done:
                    break

                self.c +=1
                # print("Timesteps", self.c)
                #print(ms)
                '''
                    Replace q_target with q_net if c%50 == 0
                '''

                if self.c % 50 == 0:
                    # self.q_target.parameters() = self.q_net.parameters()
                    self.q_target.load_state_dict(self.q_net.state_dict())
                    # self.q_target.eval()

            # print("Rewards: ", rewards)

            if e % 100 == 0:
                # pass
                test_reward = 0
                for _ in range(20):
                    state, _ = env.reset()
                    state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                    for _ in range(max_steps):
                        action = self.get_action(state, stochastic=False)
                        next_state, reward, done, log, _ = env.step(action.item())
                        next_state, reward, done, _ = torch.tensor([next_state], dtype=torch.float32).to(self.device), torch.tensor([reward], dtype=torch.float32).to(self.device), done, log
                        state = next_state
                        test_reward += reward.detach().item()
                        if done:
                            break
                total_rewards.append(test_reward/20)
                print("Epsodic reward in eval",test_reward/20)
                #print("rewards:", rewards)
                # print("\tEpisode {} \t Final Reward {:.2f} \t Average Reward: {:.2f}".format(e, rewards))


        # pass

        # END STUDENT SOLUTION
        return total_rewards



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
    max_steps = args.max_steps
    num_episodes = args.num_episodes
    num_runs = args.num_runs

    # init args, agents, and call graph_agent on the initialized agents
    # BEGIN STUDENT SOLUTION
    '''
        TODO:
            * verify if we can make the environment global
    '''
    # global env
    global env
    env = gym.make(args.env_name, max_episode_steps=max_steps)

    # device = torch.device(
    # "cuda" if torch.cuda.is_available() else
    # "mps" if torch.backends.mps.is_available() else
    # "cpu"
    # )
    device = torch.device("cpu")
    
    n_actions = env.action_space.n
    state, info = env.reset()
    n_observations = len(state)
    
    # send in action_size & state_size
    for run in range(num_runs):
        dqn = DeepQNetwork(n_observations, n_actions)
        run_reward = dqn.run(env, max_steps, num_episodes)
    env.close()

    # END STUDENT SOLUTION



if '__main__' == __name__:
    main()
