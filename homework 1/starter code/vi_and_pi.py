### MDP Value Iteration and Policy Iteration

import numpy as np
import gym
import time
from lake_envs import * # imported environments from gym
import time


np.set_printoptions(precision=3)

"""
For policy_evaluation, policy_improvement, policy_iteration and value_iteration,
the parameters P, nS, nA, gamma are defined as follows:

    P: nested dictionary
        From gym.core.Environment
        For each pair of states in [1, nS] and actions in [1, nA], P[state][action] is a
        tuple of the form (probability, nextstate, reward, terminal) where
            - probability: float
                the probability of transitioning from "state" to "nextstate" with "action"
            - nextstate: int
                denotes the state we transition to (in range [0, nS - 1])
            - reward: int
                either 0 or 1, the reward for transitioning from "state" to
                "nextstate" with "action"
            - terminal: bool
              True when "nextstate" is a terminal state (hole or goal), False otherwise
    nS: int
        number of states in the environment
    nA: int
        number of actions in the environment
    gamma: float
        Discount factor. Number in range [0, 1)
"""

def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-3):
    """Evaluate the value function from a given policy.

    Parameters
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    policy: np.array[nS]
        The policy to evaluate. Maps states to actions.
    tol: float
        Terminate policy evaluation when
            max |value_function(s) - prev_value_function(s)| < tol
    Returns
    -------
    value_function: np.ndarray[nS]
        The value function of the given policy, where value_function[s] is
        the value of state s
    """

    value_function = np.zeros(nS)
    V_new = value_function.copy()

    ############################
    # YOUR IMPLEMENTATION HERE #
  
    # stopping condition
    max_it = 1000
    it = 0
    
    # iterate until convergence
    while (it <= max_it or sum(np.abs(value_function - V_new))>tol):
        value_function = V_new.copy()
        # loop over all states
        for s in range(nS):
            # action at this state s according to current policy:
            a = policy[s]
            # load what happens from dictionary:
            result = P[s][a]
            # immediate reward (average over possible "nextstates"):
            V_new[s] = np.array(result)[:,2].mean() 
        
            # loop over possible "nextstates" (for stochastic case)
            for index in range(len(result)):
                # each scenario:
                (p,s_prime,r,T) = result[index]
                # add to get total value
                V_new[s] += gamma * p * value_function[s_prime]
        # update iteration:
        it+=1
        
    return V_new
   ############################


def policy_improvement(P, nS, nA, value_from_policy, policy, gamma=0.9):
    """Given the value function from policy improve the policy.

    Parameters
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    value_from_policy: np.ndarray
        The value calculated from the policy
    policy: np.array
        The previous policy.

    Returns
    -------
    new_policy: np.ndarray[nS]
        An array of integers. Each integer is the optimal action to take
        in that state according to the environment dynamics and the
        given value function.
    """

    new_policy = np.zeros(nS, dtype='int')

    ############################
    # YOUR IMPLEMENTATION HERE #

    # define the Q function.
    # greedy policy will pick a = argmax Q(s,a) for each s
    Q = np.zeros([nS,nA])
    
    # loop over all states
    for s in range(nS):
        
        # loop over all actions
        for a in range(nA):
            # result - load from dictionary
            result = P[s][a]
            
            # expected value (loop over possible "newstates" in result)
            for entry in range(len(result)):
                p,s_prime,r,T = result[entry]
                Q[s][a] = r
                Q[s][a] += gamma * p * value_from_policy[s_prime]
    
    # we have a full Q matrix, for each state select a that maximises the reward:
    new_policy = np.argmax(Q,axis=1)
    
    ############################
    return new_policy


def policy_iteration(P, nS, nA, gamma=0.9, tol=10e-3):
    """Runs policy iteration.

    You should call the policy_evaluation() and policy_improvement() methods to
    implement this method.

    Parameters
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    tol: float
        tol parameter used in policy_evaluation()
    Returns:
    ----------
    value_function: np.ndarray[nS]
    policy: np.ndarray[nS]
    """

    value_function = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)

    ############################
    # YOUR IMPLEMENTATION HERE #

    # iteration stopping
    max_iteration = 1000
    it = 0 
    
    new_policy= policy.copy()
    while (it <= max_iteration) or (sum(np.abs(new_policy - policy))>tol):
        
        # update policy
        policy = new_policy
        
        # call policy_evaluation
        # find value for the new policy
        value_function = policy_evaluation(P, nS, nA, policy)
        
        # use this new value to update policy (greedy manner)
        new_policy = policy_improvement(P, nS, nA, value_function, policy)
        it += 1
    ############################
    return value_function, policy
    

def value_iteration(P, nS, nA, gamma=0.9, tol=1e-3):
    """
    Learn value function and policy by using value iteration method for a given
    gamma and environment.

    Parameters:
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    tol: float
        Terminate value iteration when
            max |value_function(s) - prev_value_function(s)| < tol
    Returns:
    ----------
    value_function: np.ndarray[nS]
    policy: np.ndarray[nS]
    """

    value_function = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)
    ############################
    # YOUR IMPLEMENTATION HERE #
    
    # iterations
    idx = 0
    max_iteration = 1000
    
    V_new = value_function.copy()
    
    while (idx <= max_iteration) or (sum(np.abs(V_new-value_function))>tol):
  # loop over states
        for s in range(nS):
            # for comparing
            max_result = -10
            max_idx = 0
            
            # loop over actions
            for a in range(nA):
                result = P[s][a]
                reward = np.array(result)[:,2].mean()
                
                # add expectation over possible s_prime
                for entry in range(len(result)):
                    (p, s_prime, r, T) = result[entry]
                    reward += gamma*p*value_function[s_prime]
                    
                    # pick the reward > threshold
                    if max_result < reward:
                        
                        max_result = reward
                        max_idx = a
            V_new[s] = max_result
            policy[s] = max_idx

        idx += 1
        value_function = V_new
    ############################
    return value_function, policy
    
    
def render_single(env, policy, max_steps=100):
  """
    This function does not need to be modified
    Renders policy once on environment. Watch your agent play!

    Parameters
    ----------
    env: gym.core.Environment
      Environment to play on. Must have nS, nA, and P as
      attributes.
    Policy: np.array of shape [env.nS]
      The action to take at a given state
  """

  episode_reward = 0
  ob = env.reset()
  for t in range(max_steps):
    env.render()
    time.sleep(0.25)
    a = policy[ob]
    ob, rew, done, _ = env.step(a)
    episode_reward += rew
    if done:
      break
  env.render();
  if not done:
    print("The agent didn't reach a terminal state in {} steps.".format(max_steps))
  else:
      print("Episode reward: %f" % episode_reward)

if __name__ == "__main__":

    # comment/uncomment these lines to switch between deterministic/stochastic environments
    env = gym.make("Deterministic-4x4-FrozenLake-v0")
    #env = gym.make("Stochastic-4x4-FrozenLake-v0")

    time_start = time.clock()
    
    print("\n" + "-"*25 + "\nBeginning Policy Iteration\n" + "-"*25)
    V_pi, p_pi = policy_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
    render_single(env, p_pi, 100)
    
    time_elapsed = (time.clock() - time_start)
    print('Policy iteration computation time: '+str(time_elapsed))

#*** 

    time_start = time.clock()
    
    print("\n" + "-"*25 + "\nBeginning Value Iteration\n" + "-"*25)
    V_vi, p_vi = value_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
    render_single(env, p_vi, 100)
    
    time_elapsed = (time.clock() - time_start)
    print('Value iteration computation time: '+str(time_elapsed))