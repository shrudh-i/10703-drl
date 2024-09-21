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
    Q = np.zeros(k)

    for step in range(0,steps):
        action = 0
        epsilon_probability = np.random.uniform(0,1)
        if (epsilon_probability <= 1 - epsilon):
            action = np.argmax(Q)
        else:
            action = np.random.randint(1,k)

        reward = meanRewards[action] + np.random.normal(0,1)
        n[action] += 1
        Q[action] += (1/ n[action])*(reward - Q[action])
        expectedRewards[step] = Q[action]

    # END STUDENT SOLUTION
    return(expectedRewards)



def optimisticInitialization(value, steps, k, meanRewards, n):
    # TODO implement the optimistic initializaiton algorithm over all steps and
    # return the expected rewards across all steps
    expectedRewards = np.zeros(steps)

    # BEGIN STUDENT SOLUTION
    Q = np.zeros(k) + value
    
    for step in range(0,steps):
        action = np.argmax(Q)

        reward = meanRewards[action] + np.random.normal(0,1)
        n[action] += 1
        Q[action] += (1/ n[action])*(reward - Q[action])
        expectedRewards[step] = Q[action]
    # END STUDENT SOLUTION
    return(expectedRewards)



def ucbExploration(c, steps, k, meanRewards, n):
    # TODO implement the UCB exploration algorithm over all steps and return the
    # expected rewards across all steps, remember to pull all arms initially
    expectedRewards = np.zeros(steps)
    # BEGIN STUDENT SOLUTION
    Q = np.zeros(k) 
    
    for step in range(1,steps+1):
        action = np.argmax(Q + c*np.sqrt(np.log10(step)/(n+1e-6)))

        reward = meanRewards[action] + np.random.normal(0,1)
        n[action] += 1
        Q[action] += (1/ n[action])*(reward - Q[action])
        expectedRewards[step-1] = Q[action]
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
    iters = 1000 #was 1000
    alg_to_name = {epsilonGreedyExploration : 'Epsilon Greedy Exploration',
                   optimisticInitialization : 'Optimistic Initialization',
                   ucbExploration: 'UCB Exploration',
                   boltzmannExploration: 'Boltzmann Exploration'}
    # BEGIN STUDENT SOLUTION
    plt.figure()
    for algo, param in alg_param_list:
        expected_rewards = runExplorationAlgorithm(algo, param, iters)
        plt.plot(np.arange(1,1001), expected_rewards, label = param)
    # END STUDENT SOLUTION
    plt.legend()
    plt.show()



if __name__ == '__main__':
    # TODO call plotAlgorithms here to plot your algorithms
    np.random.seed(10003)

    # BEGIN STUDENT SOLUTION

    '''
    # EPSILON GREEDY ALGO:
    expected_returns = runExplorationAlgorithm(epsilonGreedyExploration, 0.1, 10)
    alg_param_list = np.array([(epsilonGreedyExploration,0.1)])
    plotAlgorithms(alg_param_list)
    '''

    '''
    # OPTIMISM ALGO:
    expected_returns = runExplorationAlgorithm(optimisticInitialization, 5, 10)
    alg_param_list = np.array([
                                (optimisticInitialization,0), 
                                (optimisticInitialization,1), 
                                (optimisticInitialization,2),
                                (optimisticInitialization,5),
                                (optimisticInitialization,10)])
    plotAlgorithms(alg_param_list)
    '''

    '''
    # UCB ALGO:
    expected_returns = runExplorationAlgorithm(ucbExploration, 5, 10)
    alg_param_list = np.array([
                                (ucbExploration,0), 
                                (ucbExploration,1), 
                                (ucbExploration,2),
                                (ucbExploration,5)])
    plotAlgorithms(alg_param_list)
    '''

    # END STUDENT SOLUTION
