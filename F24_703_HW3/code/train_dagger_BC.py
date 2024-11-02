import gymnasium as gym
import os
import numpy as np
import torch
from torch import nn

from modules import PolicyNet, ValueNet
from simple_network import SimpleNet
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

try:
    import wandb
except ImportError:
    wandb = None

class TrainDaggerBC:

    def __init__(self, env, model, expert_model, optimizer, states, actions, device="cpu", mode="DAgger"):
        """
        Initializes the TrainDAgger class. Creates necessary data structures.

        Args:
            env: an OpenAI Gym environment.
            model: the model to be trained.
            expert_model: the expert model that provides the expert actions.
            device: the device to be used for training.
            mode: the mode to be used for training. Either "DAgger" or "BC".

        """
        self.env = env
        self.model = model
        self.expert_model = expert_model
        self.optimizer = optimizer
        self.device = device
        #model.set_device(self.device)
        model.to(self.device)
        self.expert_model = self.expert_model.to(self.device)

        self.mode = mode

        if self.mode == "BC":
            self.states = []
            self.actions = []
            self.timesteps = []
            for trajectory in range(states.shape[0]):
                trajectory_mask = states[trajectory].sum(axis=1) != 0
                self.states.append(states[trajectory][trajectory_mask])
                self.actions.append(actions[trajectory][trajectory_mask])
                self.timesteps.append(np.arange(0, len(trajectory_mask)))
            self.states = np.concatenate(self.states, axis=0)
            self.actions = np.concatenate(self.actions, axis=0)
            self.timesteps = np.concatenate(self.timesteps, axis=0)

            self.clip_sample_range = 1
            self.actions = np.clip(self.actions, -self.clip_sample_range, self.clip_sample_range)

        else:
            self.states = None
            self.actions = None
            self.timesteps = None

    def generate_trajectory(self, env, policy):
        """Collects one rollout from the policy in an environment. The environment
        should implement the OpenAI Gym interface. A rollout ends when done=True. The
        number of states and actions should be the same, so you should not include
        the final state when done=True.

        Args:
            env: an OpenAI Gym environment.
            policy: The output of a deep neural network
            Returns:
            states: a list of states visited by the agent.
            actions: a list of actions taken by the agent. Note that these actions should never actually be trained on...
            timesteps: a list of integers, where timesteps[i] is the timestep at which states[i] was visited.
        """

        states, old_actions, timesteps, rewards = [], [], [], []

        done, trunc = False, False
        cur_state, _ = env.reset()  
        t = 0
        while (not done) and (not trunc):
            with torch.no_grad():
                p = policy(torch.from_numpy(cur_state).to(self.device).float().unsqueeze(0), torch.tensor(t).to(self.device).long().unsqueeze(0))
            a = p.cpu().numpy()[0]
            next_state, reward, done, trunc, _ = env.step(a)
            states.append(cur_state)
            old_actions.append(a)
            timesteps.append(t)
            rewards.append(reward)

            t += 1

            cur_state = next_state

        return states, old_actions, timesteps, rewards

    def call_expert_policy(self, state):
        """
        Calls the expert policy to get an action.

        Args:
            state: the current state of the environment.
        """
        # takes in a np array state and returns an np array action
        with torch.no_grad():
            state_tensor = torch.tensor(np.expand_dims(state, axis=0), dtype=torch.float32, device=self.device)
            action = self.expert_model.choose_action(state_tensor, deterministic=True).cpu().numpy()
            action = np.clip(action, -1, 1)[0]
        return action

    def update_training_data(self, num_trajectories_per_batch_collection=20):
        """
        Updates the training data by collecting trajectories from the current policy and the expert policy.

        Args:
            num_trajectories_per_batch_collection: the number of trajectories to collect from the current policy.

        NOTE: you will need to call self.generate_trajectory and self.call_expert_policy in this function.
        NOTE: you should update self.states, self.actions, and self.timesteps in this function.
        """
        # BEGIN STUDENT SOLUTION
        # self.states = self.states.tolist()
        # self.actions = self.action.tolist()
        # self.timesteps = self.timesteps.tolist()
        rewards = []

        for num in range(num_trajectories_per_batch_collection):
            states, old_actions, timesteps, reward = self.generate_trajectory(self.env, self.model)

            for i in range(len(states)):
                state = states[i]
                timestep = timesteps[i]
                expert_action = self.call_expert_policy(state)

                #aggregation
                self.states = np.append(self.states, np.array([state]), axis=0)
                # self.states = np.concatenate((self.states, state), axis=0)
                self.actions = np.append(self.actions, np.array([expert_action]), axis = 0)
                # self.actions.append(expert_action)
                self.timesteps = np.append(self.timesteps, np.array(timestep))
                # self.timesteps.append(timestep)
                rewards.append(np.sum(reward))
        # END STUDENT SOLUTION
        # self.states = np.ndarray(self.states)
        # self.action = np.ndarray(self.action)
        # self.timestep = np.ndarray(self.timestep)

        return rewards

    def generate_trajectories(self, num_trajectories_per_batch_collection=20):
        """
        Runs inference for a certain number of trajectories. Use for behavior cloning.

        Args:
            num_trajectories_per_batch_collection: the number of trajectories to collect from the current policy.
        
        NOTE: you will need to call self.generate_trajectory in this function.
        """
        # BEGIN STUDENT SOLUTION
        rewards = []

        for num_traj in range(num_trajectories_per_batch_collection):
            _, _, _, reward = self.generate_trajectory(self.env, self.model)
            rewards.append(np.sum(reward))

        rewards = np.array(rewards)
        # END STUDENT SOLUTION

        return rewards

    def train(
        self, 
        num_batch_collection_steps=0, 
        num_BC_training_steps=20000,
        num_training_steps_per_batch_collection=1000, 
        num_trajectories_per_batch_collection=20, 
        batch_size=64, 
        print_every=500, 
        save_every=10000, 
        wandb_logging=False
    ):
        """
        Train the model using BC or DAgger

        Args:
            num_batch_collection_steps: the number of times to collecta batch of trajectories from the current policy.
            num_BC_training_steps: the number of iterations to train the model using BC.
            num_training_steps_per_batch_collection: the number of times to train the model per batch collection.
            num_trajectories_per_batch_collection: the number of trajectories to collect from the current policy per batch.
            batch_size: the batch size to use for training.
            print_every: how often to print the loss during training.
            save_every: how often to save the model during training.
            wandb_logging: whether to log the training to wandb.

        NOTE: for BC, you will need to call the self.training_step function and self.generate_trajectories function.
        NOTE: for DAgger, you will need to call the self.training_step and self.update_training_data function.
        """

        # BEGIN STUDENT SOLUTION
        losses = []
        average_rewards = []
        median_rewards = []
        max_rewards = []
        if self.mode == "BC":
            for num in range(num_BC_training_steps):
                loss = self.training_step(batch_size)
                losses.append(loss)

                if (num+1) % print_every == 0 or num == 0:
                    print("Training Step", num+1, "Loss:", loss)

                # "Evaluate Reward Every" not included, Piazza says to set to 1000
                if (num+1) % 1000 == 0 or num == 0:
                    rewards = self.generate_trajectories(num_trajectories_per_batch_collection)
                    average_rewards.append(np.average(rewards))
                    median_rewards.append(np.median(rewards))
                    max_rewards.append(np.max(rewards))
                    print("Training Step", num+1, "Rewards:", "[Avg]", np.average(rewards), "[Med]", np.median(rewards), "[Max]", np.max(rewards))

        elif self.mode == "DAgger":
            # use num_batch_collection_steps & num_training_steps_per_batch_collections in DAgger only
            self.states = np.empty((0,24))
            self.actions = np.empty((0,4))
            self.timesteps = np.empty((0,1), dtype=int)
            for i in range(num_batch_collection_steps):
                rewards = self.update_training_data()
                average_rewards.append(np.average(rewards))
                median_rewards.append(np.median(rewards))
                max_rewards.append(np.max(rewards))
                print("Batch", i+1, "Rewards:", "[Avg]", np.average(rewards), "[Med]", np.median(rewards), "[Max]", np.max(rewards))

                for j in range(num_training_steps_per_batch_collection):
                    loss = self.training_step(batch_size)
                    losses.append(loss)

                    if (j+1) % print_every == 0 or j == 0:
                        print("Training Step", j+1, "Loss:", loss)

        losses = np.array(losses)

        # for plotting
        x = np.arange(len(losses))
        ys = np.array([losses])
        print(ys)
        self.plot_results(x, ys, ["loss"], "Loss", "Iterations", "Loss")

        x = np.arange(len(average_rewards))
        if self.mode == "BC":
            x = x * 1000
        ys = np.array([average_rewards, median_rewards, max_rewards])
        self.plot_results(x, ys, ["Average", "Median", "Max"], "Rewards", "Iterations", "Rewards")
        # END STUDENT SOLUTION

        return losses

    def training_step(self, batch_size):
        """
        Simple training step implementation

        Args:
            batch_size: the batch size to use for training.
        """
        states, actions, timesteps = self.get_training_batch(batch_size=batch_size)

        states = states.to(self.device)
        actions = actions.to(self.device)
        timesteps = timesteps.to(self.device)

        loss_fn = nn.MSELoss()
        self.optimizer.zero_grad()
        predicted_actions = self.model(states, timesteps)
        loss = loss_fn(predicted_actions, actions)
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().item()

    def get_training_batch(self, batch_size=64):
        """
        get a training batch

        Args:
            batch_size: the batch size to use for training.
        """
        # get random states, actions, and timesteps
        indices = np.random.choice(len(self.states), size=batch_size, replace=False)
        states = torch.tensor(self.states[indices], device=self.device).float()
        actions = torch.tensor(self.actions[indices], device=self.device).float()
        timesteps = torch.tensor(self.timesteps[indices], device=self.device)
            
        
        return states, actions, timesteps
    
    def plot_results(self, x, ys, labels, title, x_axis_label, y_axis_label):
        plt.figure()
        for i in range(len(ys)):
            plt.plot(x, ys[i], label=labels[i])
        plt.legend()
        plt.xlabel(x_axis_label)
        plt.ylabel(y_axis_label)
        plt.title(title)
        plt.show()


def run_training():
    """
    Simple Run Training Function
    """

    env = gym.make('BipedalWalker-v3') # , render_mode="rgb_array"

    model_name = "super_expert_PPO_model"
    expert_model = PolicyNet(24, 4)
    model_weights = torch.load(f"data/models/{model_name}.pt", map_location=torch.device('cpu'))
    expert_model.load_state_dict(model_weights["PolicyNet"])

    states_path = "data/states_BC.pkl"
    actions_path = "data/actions_BC.pkl"

    with open(states_path, "rb") as f:
        states = pickle.load(f)
    with open(actions_path, "rb") as f:
        actions = pickle.load(f)

    # BEGIN STUDENT SOLUTION
    hidden_layer_dimensions = 128
    max_episode_length = 1600
    model = SimpleNet(24, 4, hidden_layer_dimensions, max_episode_length)

    learning_rate = 0.0001
    weight_decay = 0.0001
    optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate, weight_decay = weight_decay)

    trainBC = TrainDaggerBC(env, model, expert_model, optimizer, states, actions, "cpu", "BC")
    trainDagger = TrainDaggerBC(env, model, expert_model, optimizer, states, actions, "cpu", "DAgger")
    
    #losses = trainBC.train()
    losses = trainDagger.train(num_batch_collection_steps=20, num_training_steps_per_batch_collection=1000, batch_size=128)
    # END STUDENT SOLUTION

if __name__ == "__main__":
    run_training()