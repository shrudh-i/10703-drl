#! python3

import numpy as np
import matplotlib.pyplot as plt



# Use the run exploration algorithm we have provided to get you
# started, this returns the expected rewards after running an exploration 
# algorithm in the K-Armed Bandits problem. We have already specified a number
# of parameters specific to the 10-armed testbed for guidance.



def runExplorationAlgorithm(explorationAlgorithm, param, iters):
    cumulativeRewards = []
    for i in range(iters):
        # number of time steps
        t = 1000
        # number of arms, 10 in this instance
        k = 10
        # mean reward across each of the K arms
        # sample the actual rewards from a normal distribution with mean of meanRewards and standard deviation of 1
        meanRewards = np.random.normal(1,1,k)
        # counts for each arm
        n = np.zeros(k)
        # extract expected rewards by running specified exploration algorithm with the parameters above
        # param is the different, specific parameter for each exploration algorithm
        # this would be epsilon for epsilon greedy, initial values for optimistic intialization, c for UCB, and temperature for Boltmann
        currentRewards = explorationAlgorithm(param, t, k, meanRewards, n)
        cumulativeRewards.append(currentRewards)
    # calculate average rewards across each iteration to produce expected rewards
    expectedRewards = np.mean(cumulativeRewards, axis=0)
    return expectedRewards



def epsilonGreedyExploration(epsilon, steps, k, meanRewards, n):
    # TODO implement the epsilong greedy algorithm over all steps and return
    # the expected rewards across all steps
    expectedRewards = np.zeros(steps)
    # BEGIN STUDENT SOLUTION
    # END STUDENT SOLUTION
    return(expectedRewards)



def optimisticInitialization(value, steps, k, meanRewards, n):
    # TODO implement the optimistic initializaiton algorithm over all steps and
    # return the expected rewards across all steps
    expectedRewards = np.zeros(steps)
    # BEGIN STUDENT SOLUTION
    # END STUDENT SOLUTION
    return(expectedRewards)



def ucbExploration(c, steps, k, meanRewards, n):
    # TODO implement the UCB exploration algorithm over all steps and return the
    # expected rewards across all steps, remember to pull all arms initially
    expectedRewards = np.zeros(steps)
    # BEGIN STUDENT SOLUTION
    # END STUDENT SOLUTION
    return(expectedRewards)



def boltzmannExploration(temperature, steps, k, meanRewards, n):
    # TODO implement the Boltzmann Exploration algorithm over all steps and
    # return the expected rewards across all steps
    expectedRewards = np.zeros(steps)
    # BEGIN STUDENT SOLUTION
    # END STUDENT SOLUTION
    return(expectedRewards)



# plot template
def plotAlgorithms(alg_param_list):
    # TODO given a list of (algorithm, parameter) tuples, make a graph that
    # plots the expectedRewards of running that algorithm with those parameters
    # iters times using runExplorationAlgorithm plot all data on the same plot
    # include correct labels on your plot
    iters = 1000
    alg_to_name = {epsilonGreedyExploration : 'Epsilon Greedy Exploration',
                   optimisticInitialization : 'Optimistic Initialization',
                   ucbExploration: 'UCB Exploration',
                   boltzmannExploration: 'Boltzmann Exploration'}
    # BEGIN STUDENT SOLUTION
    # END STUDENT SOLUTION
    pass



if __name__ == '__main__':
    # TODO call plotAlgorithms here to plot your algorithms
    np.random.seed(10003)

    # BEGIN STUDENT SOLUTION
    # END STUDENT SOLUTION
    pass
