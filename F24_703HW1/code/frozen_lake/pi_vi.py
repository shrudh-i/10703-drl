#! python3

import numpy as np
import matplotlib.pyplot as plt
import gymnasium

import lake_info



def value_func_to_policy(env, gamma, value_func):
    '''
    Outputs a policy given a value function.

    Parameters
    ----------
    env: gymnasium.core.Environment
        The environment to compute the policy for.
    gamma: float
        Discount factor, must be in range [0, 1).
    value_func: np.ndarray
        The current value function estimate.

    Returns
    -------
    np.ndarray
        An array of integers. Each integer is the optimal action to take in
        that state according to the environment dynamics and the given value
        function.
    '''
    policy = np.zeros(env.observation_space.n, dtype='int')
    # BEGIN STUDENT SOLUTION
    for s in range(env.observation_space.n):
        rewards = np.zeros(env.action_space.n)
        for a in range(env.action_space.n):
            trans_dyn = env.P[s][a]
            for p, s_, r, _ in trans_dyn:
                rewards[a] += p*(r + gamma*value_func[s_])
        policy[s] = np.argmax(rewards)
    # END STUDENT SOLUTION
    return(policy)



def evaluate_policy_sync(env, value_func, gamma, policy, max_iters=int(1e3), tol=1e-3):
    '''
    Performs policy evaluation.

    Parameters
    ----------
    env: gymnasium.core.Environment
        The environment to compute value iteration for.
    value_func: np.ndarray
        The current value function estimate.
    gamma: float
        Discount factor, must be in range [0, 1).
    policy: np.ndarray
        The policy to evaluate, maps states to actions.
    max_iters: int
        The maximum number of iterations to run before stopping.
    tol: float
        Determines when value function has converged.

    Returns
    -------
    (np.ndarray, int)
        The value for the given policy and the number of iterations the value
        function took to converge.
    '''
    # BEGIN STUDENT SOLUTION
    for i in range(max_iters):
        delta = 0
        value_func_k1 = np.zeros(env.observation_space.n)
        for s in range(env.observation_space.n):
            v = value_func[s]
            a = policy[s]
            trans_dyn = env.P[s][a]
            value = 0
            for p, s_, r, _ in trans_dyn:
                value += p*(r + gamma*value_func[s_])
        
            value_func_k1[s] = value
            delta = max(delta, abs(v-value))
        value_func = value_func_k1.copy()

        if delta < tol:
            break

    # END STUDENT SOLUTION
    return(value_func, i)



def evaluate_policy_async_ordered(env, value_func, gamma, policy, max_iters=int(1e3), tol=1e-3):
    '''
    Performs policy evaluation.

    Evaluates the value of a given policy by asynchronous DP.
    Updates states in their 1-N order.

    Parameters
    ----------
    env: gymnasium.core.Environment
        The environment to compute value iteration for.
    value_func: np.ndarray
        The current value function estimate.
    gamma: float
        Discount factor, must be in range [0, 1).
    policy: np.ndarray
        The policy to evaluate, maps states to actions.
    max_iters: int
        The maximum number of iterations to run before stopping.
    tol: float
        Determines when value function has converged.

    Returns
    -------
    (np.ndarray, int)
        The value for the given policy and the number of iterations the value
        function took to converge.
    '''
    # BEGIN STUDENT SOLUTION
    for i in range(max_iters):
        delta = 0
        for s in range(env.observation_space.n):
            v = value_func[s]
            a = policy[s]
            trans_dyn = env.P[s][a]
            value = 0
            for p, s_, r, _ in trans_dyn:
                value += p*(r + gamma*value_func[s_])
    
            value_func[s] = value
            delta = max(delta, abs(v-value))

        if delta < tol:
            break
    # END STUDENT SOLUTION
    return(value_func, i)



def evaluate_policy_async_randperm(env, value_func, gamma, policy, max_iters=int(1e3), tol=1e-3):
    '''
    Performs policy evaluation.

    Evaluates the value of a policy. Updates states by randomly sampling index
    order permutations.

    Parameters
    ----------
    env: gymnasium.core.Environment
        The environment to compute value iteration for.
    value_func: np.ndarray
        The current value function estimate.
    gamma: float
        Discount factor, must be in range [0, 1).
    policy: np.ndarray
        The policy to evaluate, maps states to actions.
    max_iters: int
        The maximum number of iterations to run before stopping.
    tol: float
        Determines when value function has converged.

    Returns
    -------
    (np.ndarray, int)
        The value for the given policy and the number of iterations the value
        function took to converge.
    '''
    # BEGIN STUDENT SOLUTION
    for i in range(max_iters):
        delta = 0
        for s in np.random.permutation(env.observation_space.n):
            v = value_func[s]
            a = policy[s]
            trans_dyn = env.P[s][a]
            value = 0
            for p, s_, r, _ in trans_dyn:
                value += p*(r + gamma*value_func[s_])
            
            value_func[s] = value
            delta = max(delta, abs(v-value))

            if delta < tol:
                break
    # END STUDENT SOLUTION
    return(value_func, i)



def improve_policy(env, gamma, value_func, policy):
    '''
    Performs policy improvement.

    Given a policy and value function, improves the policy.

    Parameters
    ----------
    env: gymnasium.core.Environment
        The environment to compute value iteration for.
    gamma: float
        Discount factor, must be in range [0, 1).
    value_func: np.ndarray
        The current value function estimate.
    policy: np.ndarray
        The policy to improve, maps states to actions.

    Returns
    -------
    (np.ndarray, bool)
        Returns the new policy and whether the policy changed.
    '''
    policy_changed = False
    # BEGIN STUDENT SOLUTION
    for s in range(env.observation_space.n):
        old_action = policy[s]
        rewards = np.zeros(env.action_space.n)
        for a in range(env.action_space.n):
            trans_dyn = env.P[s][a]
            for p, s_, r, _ in trans_dyn:
                rewards[a] += p*(r + gamma*value_func[s_])

        policy[s] = np.argmax(rewards)

        if old_action != policy[s]:
            policy_changed = True

    # END STUDENT SOLUTION
    return(policy, policy_changed)



def policy_iteration_sync(env, gamma, max_iters=int(1e3), tol=1e-3):
    '''
    Runs policy iteration.

    You should use the improve_policy() and evaluate_policy_sync() methods to
    implement this method.

    Parameters
    ----------
    env: gymnasium.core.Environment
        The environment to compute value iteration for.
    gamma: float
        Discount factor, must be in range [0, 1).
    max_iters: int
        The maximum number of iterations to run before stopping.
    tol: float
        Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
        Returns optimal policy, value function, number of policy improvement
        iterations, and number of policy evaluation iterations.
    '''
    policy = np.zeros(env.observation_space.n, dtype='int')
    value_func = np.zeros(env.observation_space.n)
    pi_steps, pe_steps = 0, 0
    # BEGIN STUDENT SOLUTION
    policy_changed = True

    while policy_changed:
        value_func, num_iters = evaluate_policy_sync(env, value_func, gamma, policy, max_iters, tol)
        pe_steps += num_iters
        policy, policy_changed = improve_policy(env, gamma, value_func, policy)
        pi_steps += 1
    # END STUDENT SOLUTION
    return(policy, value_func, pi_steps, pe_steps)



def policy_iteration_async_ordered(env, gamma, max_iters=int(1e3), tol=1e-3):
    '''
    Runs policy iteration.

    You should use the improve_policy and evaluate_policy_async_ordered methods
    to implement this method.

    Parameters
    ----------
    env: gymnasium.core.Environment
        The environment to compute value iteration for.
    gamma: float
        Discount factor, must be in range [0, 1).
    max_iters: int
        The maximum number of iterations to run before stopping.
    tol: float
        Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
        Returns optimal policy, value function, number of policy improvement
        iterations, and number of policy evaluation iterations.
    '''
    policy = np.zeros(env.observation_space.n, dtype='int')
    value_func = np.zeros(env.observation_space.n)
    pi_steps, pe_steps = 0, 0
    # BEGIN STUDENT SOLUTION
    policy_changed = True
    while policy_changed:
        value_func, num_iters = evaluate_policy_async_ordered(env, value_func, gamma, policy, max_iters, tol)
        pe_steps += num_iters
        policy, policy_changed = improve_policy(env, gamma, value_func, policy)
        pi_steps += 1
    # END STUDENT SOLUTION
    return(policy, value_func, pi_steps, pe_steps)



def policy_iteration_async_randperm(env, gamma, max_iters=int(1e3), tol=1e-3):
    '''
    Runs policy iteration.

    You should use the improve_policy and evaluate_policy_async_randperm methods
    to implement this method.

    Parameters
    ----------
    env: gymnasium.core.Environment
        The environment to compute value iteration for.
    gamma: float
        Discount factor, must be in range [0, 1).
    max_iters: int
        The maximum number of iterations to run before stopping.
    tol: float
        Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
        Returns optimal policy, value function, number of policy improvement
        iterations, and number of policy evaluation iterations.
    '''
    policy = np.zeros(env.observation_space.n, dtype='int')
    value_func = np.zeros(env.observation_space.n)
    pi_steps, pe_steps = 0, 0
    # BEGIN STUDENT SOLUTION
    policy_changed = True 
    while policy_changed:
        value_func, num_iters = evaluate_policy_async_randperm(env, value_func, gamma, policy, max_iters, tol)
        pe_steps += num_iters
        policy, policy_changed = improve_policy(env, gamma, value_func, policy)
        pi_steps += 1
    # END STUDENT SOLUTION
    return(policy, value_func, pi_steps, pe_steps)



def value_iteration_sync(env, gamma, max_iters=int(1e3), tol=1e-3):
    '''
    Runs value iteration for a given gamma and environment.

    Parameters
    ----------
    env: gymnasium.core.Environment
        The environment to compute value iteration for.
    gamma: float
        Discount factor, must be in range [0, 1).
    max_iters: int
        The maximum number of iterations to run before stopping.
    tol: float
        Determines when value function has converged.

    Returns
    -------
    (np.ndarray, iteration)
        Returns the value function, and the number of iterations it took to
        converge.
    '''
    value_func = np.zeros(env.observation_space.n)
    # BEGIN STUDENT SOLUTION
    for i in range(max_iters):
        delta = 0
        value_func_k1 = np.zeros(env.observation_space.n)
        
        for s in range(env.observation_space.n):
            v = value_func[s]

            value = np.zeros(env.action_space.n)
            for a in range(env.action_space.n):
                trans_dyn = env.P[s][a]
                for p, s_, r, _ in trans_dyn:
                    value[a] += p*(r + gamma*value_func[s_])

            new_v = max(value)
            value_func_k1[s] = new_v
            delta = max(delta, abs(v-new_v))
        value_func = value_func_k1.copy()

        if delta < tol:
            break

    # END STUDENT SOLUTION
    return(value_func, i)



def value_iteration_async_ordered(env, gamma, max_iters=int(1e3), tol=1e-3):
    '''
    Runs value iteration for a given gamma and environment.
    Updates states in their 1-N order.

    Parameters
    ----------
    env: gymnasium.core.Environment
        The environment to compute value iteration for.
    gamma: float
        Discount factor, must be in range [0, 1).
    max_iters: int
        The maximum number of iterations to run before stopping.
    tol: float
        Determines when value function has converged.

    Returns
    -------
    (np.ndarray, iteration)
        Returns the value function, and the number of iterations it took to
        converge.
    '''
    value_func = np.zeros(env.observation_space.n)
    # BEGIN STUDENT SOLUTION
    for i in range(max_iters):
        delta = 0
        
        for s in range(env.observation_space.n):
            v = value_func[s]

            value = np.zeros(env.action_space.n)
            for a in range(env.action_space.n):
                trans_dyn = env.P[s][a]
                for p, s_, r, _ in trans_dyn:
                    value[a] += p*(r + gamma*value_func[s_])

            new_v = max(value)
            value_func[s] = new_v
            delta = max(delta, abs(v-new_v))

        if delta < tol:
            break
    # END STUDENT SOLUTION
    return(value_func, i)



def value_iteration_async_randperm(env, gamma, max_iters=int(1e3), tol=1e-3):
    '''
    Runs value iteration for a given gamma and environment.
    Updates states by randomly sampling index order permutations.

    Parameters
    ----------
    env: gymnasium.core.Environment
        The environment to compute value iteration for.
    gamma: float
        Discount factor, must be in range [0, 1).
    max_iters: int
        The maximum number of iterations to run before stopping.
    tol: float
        Determines when value function has converged.

    Returns
    -------
    (np.ndarray, iteration)
        Returns the value function, and the number of iterations it took to
        converge.
    '''
    value_func = np.zeros(env.observation_space.n)
    # BEGIN STUDENT SOLUTION
    for i in range(max_iters):
        delta = 0
        
        for s in np.random.permutation(env.observation_space.n):
            v = value_func[s]

            value = np.zeros(env.action_space.n)
            for a in range(env.action_space.n):
                trans_dyn = env.P[s][a]
                for p, s_, r, _ in trans_dyn:
                    value[a] += p*(r + gamma*value_func[s_])

            new_v = max(value)
            value_func[s] = new_v
            delta = max(delta, abs(v-new_v))

        if delta < tol:
            break
    # END STUDENT SOLUTION
    return(value_func, i)



def value_iteration_async_custom(env, gamma, max_iters=int(1e3), tol=1e-3):
    '''
    Runs value iteration for a given gamma and environment.
    Updates states by student-defined heuristic.

    Parameters
    ----------
    env: gymnasium.core.Environment
        The environment to compute value iteration for.
    gamma: float
        Discount factor, must be in range [0, 1).
    max_iters: int
        The maximum number of iterations to run before stopping.
    tol: float
        Determines when value function has converged.

    Returns
    -------
    (np.ndarray, iteration)
        Returns the value function, and the number of iterations it took to
        converge.
    '''
    value_func = np.zeros(env.observation_space.n)

    # BEGIN STUDENT SOLUTION

    def get_manhattan(n, start=(0,0)):
        '''
        function to get manhattan distance from a starting point
        '''
        matrix = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                matrix[i][j] = np.sqrt((i-start[0])**2+(j-start[1])**2)
        return matrix

    maps = {
        '4x4': (1, 1),
        '8x8': (7, 1)
    }

    goal = maps[env.map_name]
    print(goal)
    ordered_states = np.argsort(get_manhattan(int(np.sqrt(env.observation_space.n)), start=goal).flatten())

    for i in range(max_iters):
        delta = 0
        
        for s in ordered_states:
            v = value_func[s]

            value = np.zeros(env.action_space.n)
            for a in range(env.action_space.n):
                trans_dyn = env.P[s][a]
                for p, s_, r, _ in trans_dyn:
                    value[a] += p*(r + gamma*value_func[s_])

            new_v = max(value)
            value_func[s] = new_v
            delta = max(delta, abs(v-new_v))

        if delta < tol:
            break
            
    # END STUDENT SOLUTION
    return(value_func, i)



# Here we provide some helper functions for your convinience.

def display_policy_letters(env, policy):
    '''
    Displays a policy as an array of letters.

    Parameters
    ----------
    env: gymnasium.core.Environment
        The environment to display the policy for.
    policy: np.ndarray
        The policy to display, maps states to actions.
    '''
    policy_letters = []
    for l in policy:
        policy_letters.append(lake_info.actions_to_names[l][0])

    policy_letters = np.array(policy_letters).reshape(env.unwrapped.nrow, env.unwrapped.ncol)

    for row in range(env.unwrapped.nrow):
        print(''.join(policy_letters[row, :]))



def value_func_heatmap(env, value_func):
    '''
    Visualize a policy as a heatmap.

    Parameters
    ----------
    env: gymnasium.core.Environment
        The environment to display the policy for.
    value_func: np.ndarray
        The current value function estimate.
    '''
    fig, ax = plt.subplots(figsize=(7,6))

    # Reshape value_func to match the environment dimensions
    heatmap_data = np.reshape(value_func, [env.unwrapped.nrow, env.unwrapped.ncol])

    # Create a heatmap using Matplotlib
    cax = ax.matshow(heatmap_data, cmap='GnBu_r')

    # Set ticks and labels
    ax.set_yticks(np.arange(0, env.unwrapped.nrow))
    ax.set_xticks(np.arange(0, env.unwrapped.ncol))
    ax.set_yticklabels(np.arange(1, env.unwrapped.nrow + 1)[::-1])
    ax.set_xticklabels(np.arange(1, env.unwrapped.ncol + 1))

    # Display the colorbar
    cbar = plt.colorbar(cax)

    plt.show()



if __name__ == '__main__':
    np.random.seed(10003)
    maps = lake_info.maps
    gamma = 0.9

    # START STUDENT SOLUTION
    for map_name, map in maps.items():
        env = gymnasium.make('FrozenLake-v1', desc=map, map_name = map_name, is_slippery=False)
        env.map_name = map_name

        '''
        Synchronous Policy Iteration
        '''
        policy, value_func, pi_steps, pe_steps = policy_iteration_sync(env, gamma, max_iters=int(1e3), tol=1e-3)
        print("policy iteration sync")
        print("improvement steps:{}\n evaluation steps:{}".format(pi_steps, pe_steps))
        display_policy_letters(env, policy)
        value_func_heatmap(env, value_func)

        '''
        Synchronous Value Iteration
        '''
        value_func, iters = value_iteration_sync(env, gamma, max_iters=int(1e3), tol=1e-3)
        policy = value_func_to_policy(env, gamma, value_func)
        print("value iteration sync")
        print("steps:{}".format(iters))
        display_policy_letters(env, policy)
        value_func_heatmap(env, value_func)

        '''
        Asynchronous Policy Iteration
        '''
        policy, value_func, pi_steps, pe_steps = policy_iteration_async_ordered(env, gamma, max_iters=int(1e3), tol=1e-3)
        print("policy iteration async ordered")
        print("improvement steps:{}\n evaluation steps:{}".format(pi_steps, pe_steps))

        '''
        Asynchronous Value Iteration
        '''
        iter1, iter2 = [], []
        for i in range(10):
            _, _, pi_steps, pe_steps = policy_iteration_async_randperm(env, gamma, max_iters=int(1e3), tol=1e-3)
            iter1.append(pi_steps)
            iter2.append(pe_steps)
        pi_steps = np.mean(iter1)
        pe_steps = np.mean(iter2)
        print("policy iteration async randperm")
        print("average over 10 trials")
        print("improvement steps:{}\n evaluation steps:{}".format(pi_steps, pe_steps))

        '''
        Aysnchronous Value Iteration: Ordered
        '''
        value_func, iters = value_iteration_async_ordered(env, gamma, max_iters=int(1e3), tol=1e-3)
        print("value iteration async ordered")
        print("steps:{}".format(iters))

        '''
        Aysnchronous Value Iteration: Randperm
        '''
        iter = []
        for i in range(10):
            value_func, iters = value_iteration_async_randperm(env, gamma, max_iters=int(1e3), tol=1e-3)
            iter.append(iters) 
        iters = np.mean(iters) 
        print("value iteration async randperm")
        print("average over 10 trials")
        print("steps:{}".format(iters))

        
        '''
        Aysnchronous Value Iteration: Custom
        '''
        value_func, iters = value_iteration_async_custom(env, gamma, max_iters=int(1e3), tol=1e-3)
        print("value iteration async custom")
        value_func_heatmap(env, value_func)
        print("steps:", format(iters))
