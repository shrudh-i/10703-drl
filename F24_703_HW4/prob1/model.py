import os
import numpy as np
import torch
import torch.nn as nn
import operator
from functools import reduce
from utils.util import ZFilter

HIDDEN1_UNITS = 400
HIDDEN2_UNITS = 400
HIDDEN3_UNITS = 400

import logging
log = logging.getLogger('root')


class PENN(nn.Module):
    """
    (P)robabilistic (E)nsemble of (N)eural (N)etworks
    """

    def __init__(self, num_nets, state_dim, action_dim, learning_rate, device=None):
        """
        :param num_nets: number of networks in the ensemble
        :param state_dim: state dimension
        :param action_dim: action dimension
        :param learning_rate:
        """

        super().__init__()
        self.num_nets = num_nets
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        # Log variance bounds
        self.max_logvar = torch.tensor(-3 * np.ones([1, self.state_dim]), dtype=torch.float, device=self.device)
        self.min_logvar = torch.tensor(-7 * np.ones([1, self.state_dim]), dtype=torch.float, device=self.device)

        # Create or load networks
        self.networks = nn.ModuleList([self.create_network(n) for n in range(self.num_nets)]).to(device=self.device)
        self.opt = torch.optim.Adam(self.networks.parameters(), lr=learning_rate)

    def forward(self, inputs):
        if not torch.is_tensor(inputs):
            inputs = torch.tensor(inputs, device=self.device, dtype=torch.float)
        return [self.get_output(self.networks[i](inputs)) for i in range(self.num_nets)]

    def get_output(self, output):
        """
        Argument:
          output: the raw output of a single ensemble member
        Return:
          mean and log variance
        """
        mean = output[:, 0:self.state_dim]
        raw_v = output[:, self.state_dim:]
        logvar = self.max_logvar - nn.functional.softplus(self.max_logvar - raw_v)
        logvar = self.min_logvar + nn.functional.softplus(logvar - self.min_logvar)
        return mean, logvar

    def get_loss(self, targ, mean, logvar):
        # TODO: write your code here

        var = torch.exp(logvar)
        mean = mean.detach().numpy()
        var = var.detach().numpy()
        print(f"type of mean size: {type(mean.size())}")
        print(f"type of var: {type(var)}")
        print(f"type of target: {type(targ)}")
        
        loss_fn = nn.GaussianNLLLoss()
        nll_loss = loss_fn(mean, targ, var)
        exit(0)

        return loss_fn(mean, targ, var)

        # return loss
        raise NotImplementedError

    def create_network(self, n):
        layer_sizes = [self.state_dim + self.action_dim, HIDDEN1_UNITS, HIDDEN2_UNITS, HIDDEN3_UNITS]
        layers = reduce(operator.add,
                        [[nn.Linear(a, b), nn.ReLU()]
                         for a, b in zip(layer_sizes[0:-1], layer_sizes[1:])])
        layers += [nn.Linear(layer_sizes[-1], 2 * self.state_dim)]
        return nn.Sequential(*layers)

    def train_model(self, inputs, targets, batch_size=128, num_train_itrs=5):
        """
        Training the Probabilistic Ensemble (Algorithm 2)
        Argument:
          inputs: state and action inputs. Assumes that inputs are standardized.
          targets: resulting states
        Return:
            List containing the average loss of all the networks at each train iteration

        """
        # TODO: write your code here
        for _ in range(1, num_train_itrs):
            # Sample a minibatch
            batchIndex = torch.randint(0, len(inputs), (batch_size,))
            minibatch_inputs = inputs[batchIndex]
            minibatch_targets = targets[batchIndex]

            # Forward pass:  returns a list of both networks
            self.opt.zero_grad()
            print(f"minibatch size {len(self.forward(minibatch_inputs))}")

            total_loss = 0

            # TODO: train each network with the neg log likelihood
            for i in range(self.num_nets):
                # print(f"this is the network: {i+1}")
                pred_mean, pred_logvar = self.forward(minibatch_inputs)[i]
                # print(f"pred_mean shape: {pred_mean.shape}")
                # print(f"pred_logvar shape: {pred_logvar.shape}")
                # print(f"minibatch_targets shape: {minibatch_targets.shape}")
                
            
                # Calculate the loss
                loss = self.get_loss(minibatch_targets, pred_mean, pred_logvar)
                print(f"this is the loss: {loss}")
                exit(0)
                total_loss += loss

            # TODO: Average loss for current minibatch 
            # avg_loss = total_loss / self.num_nets

            # Backprop and update model params

            total_loss.backward()
            # avg_loss.backward()
            self.opt.step()

        raise NotImplementedError