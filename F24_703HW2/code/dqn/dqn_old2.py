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



Transition = collections.namedtuple("Transition", field_names=['state', 'action', 'reward', 'next_state'])

class ReplayMemory():
    def __init__(self, memory_size, batch_size):
        # define init params
        # use collections.deque
        # BEGIN STUDENT SOLUTION
        self.memory = collections.deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.selected_indices = set()
        # END STUDENT SOLUTION

    def sample(self):
        return random.sample(self.memory, self.batch_size)
        # Convert deque to a list
        memory_list = list(self.memory)
        
        # Get available indices that haven't been selected
        available_indices = list(set(range(len(memory_list))) - self.selected_indices)

        if len(available_indices) < self.batch_size:
            raise ValueError("Not enough elements left to sample.")

        # Sample without replacement from available indices
        sampled_indices = random.sample(available_indices, self.batch_size)

        # Mark these indices as selected
        self.selected_indices.update(sampled_indices)

        # Return the sampled elements
        return [memory_list[i] for i in sampled_indices]


    # def sample_batch(self):
        # randomly chooses from the collections.deque
        # BEGIN STUDENT SOLUTION
        # self.replay_buffer.sample
        # samples = np.random.choice(len(self.memory), self.batch_size, replace=True)
        # batch = zip(*[self.memory[i] for i in samples])
        # batch = Transition(*zip(*t))
        # return batch
        # END STUDENT SOLUTION


    def append(self, transition):
        # append to the collections.deque
        # BEGIN STUDENT SOLUTION
        self.memory.append(np.array(transition, dtype=object))
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

        # self.loss = nn.functional.mse_loss()

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
        self.q_target = q_net_init() #copy.deepcopy(self.q_omega)
        self.q_target.load_state_dict(self.q_omega.state_dict()) 

        self.optimizer = optim.Adam(self.q_omega.parameters(), lr_q_net)
        # END STUDENT SOLUTION

    def burn_traj(self, env):
        state, _ = env.reset()
        for i in range(self.burn_in):
            action = env.action_space.sample()#np.random.choice([0,1])

            next_state, reward, done, _, _ = env.step(action)

            self.replay_buffer.append(Transition(torch.from_numpy(state).float(), action, reward, next_state))
            if done:
                state, _ = env.reset()
            else:
                state = next_state


    def forward(self, state):
        return (self.q_omega(state), self.q_target(state))


    def get_action(self, state, stochastic):
        # if stochastic, sample using epsilon greedy, else get the argmax
        # BEGIN STUDENT SOLUTION
        # print(state)
        # print(torch.from_numpy(state).float())
        with torch.no_grad():
            omega_val, target_val = self.forward(torch.from_numpy(state))
        if stochastic:
            # e-greedy
            sample = random.random()
            if sample > self.epsilon:
                # arg-max
                action = torch.argmax(omega_val)
            else:
                action = torch.randint(0,1, (1,))

        else:
            # arg-max
            action = torch.argmax(target_val)
        # END STUDENT SOLUTION
        return action.item()



    def train(self):
        # train the agent using the replay buffer
        # BEGIN STUDENT SOLUTION
        # minibatch= self.replay_buffer.sample_batch()
        trans = self.replay_buffer.sample()
        minibatch = Transition(*zip(*trans))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          minibatch.next_state)), dtype=torch.bool)
        # print(np.shape(minibatch.next_state))
        non_final_next_states = np.array([s for s in minibatch.next_state if s is not None])
        # print(np.shape(non_final_next_states))

        state_batch = torch.tensor(np.array(minibatch.state))
        # print("Minibatch state", np.shape(state_batch))
        # print(minibatch.action)
        
        action_batch = torch.tensor(minibatch.action)
        # print(np.shape(action_batch))

        reward_batch = torch.tensor(minibatch.reward)
        # print(state_batch)

        state_action_values = self.q_omega(state_batch).gather(1, action_batch.unsqueeze(1))

        next_state_values = torch.zeros(self.replay_buffer.batch_size)

        with torch.no_grad():
            next_state_values[non_final_mask] = self.q_target(torch.from_numpy(non_final_next_states)).max(1).values
        
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        
        # print("state action values",state_action_values.squeeze(1))
        # print("expected state ",expected_state_action_values)
        # loss_omega = self.loss(state_action_values.squeeze(1), expected_state_action_values)
        loss_omega = nn.functional.mse_loss(state_action_values.squeeze(1), expected_state_action_values)

        self.optimizer.zero_grad()
        loss_omega.backward()

        self.optimizer.step()

        # print(minibatch.state)
        # print("Next state with final state", minibatch.next_state)
        # non_final_mask = tuple(map(lambda s: s is not None, minibatch.next_state))
        # non_final_next_states = np.concatenate([s for s in minibatch.next_state if s is not None])
        # print("State",np.concatenate(minibatch.state))
        # print("Reward",minibatch.reward)
        # print("Action",minibatch.action)
        # print("Next state without final state", non_final_next_states)
        # print(minibatch)
        # END STUDENT SOLUTION
        


    def run(self, env, max_steps, num_episodes, train, init_buffer):
        total_rewards = []

        # initialize replay buffer
        # run the agent through the environment num_episodes times for at most max steps
        # BEGIN STUDENT SOLUTION
        self.burn_traj(env)
        c = 0
        for e in range(num_episodes):
            state, _ = env.reset()
            for t in range(max_steps):
                #get the action
                action = self.get_action(state,stochastic=True)
                next_state, reward, done, _, _ = env.step(action)
                
                if done:
                    next_state = None
                else:
                    state = next_state

                self.replay_buffer.append(Transition(torch.from_numpy(state).float(), action, reward, next_state))

                if train:
                    self.train()

                c = c + 1

                #fix ya policy, son.
                if c % 50 == 0:
                    # deepcopy
                    self.q_target.load_state_dict(self.q_omega.state_dict())

                if done:
                    break

            if (e+1) % 100 == 0:
                cumulative_reward = 0
                for _ in range(20):
                    state, _ = env.reset()
                    for i in range(max_steps):
                        action = self.get_action(state, stochastic=False)
                        next_state, reward, done, _, _ = env.step(action)
                        cumulative_reward = cumulative_reward + reward
                        if done:
                            break
                total_rewards.append(cumulative_reward/20)
                print("Epsodic reward in eval",cumulative_reward/20)
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
    env_name = args.env_name
    num_runs =args.num_runs
    num_episodes = args.num_episodes
    max_steps = args.max_steps
    # init args, agents, and call graph_agent on the initialized agents
    # BEGIN STUDENT SOLUTION
    env = gym.make(env_name)

    run_total_rewards = np.zeros((num_runs,int(num_episodes/100)),dtype=object)

    for run in range(num_runs):
        q_net = DeepQNetwork(env.observation_space.shape[0], env.action_space.n)
        rewards = q_net.run(env, max_steps, num_episodes, 1, 0)
        run_total_rewards[run] = rewards

    graph_agents("I dunno", 0, 0, max_steps= max_steps, num_episodes = num_episodes, total_rewards=run_total_rewards)

    # END STUDENT SOLUTION



if '__main__' == __name__:
    main()
