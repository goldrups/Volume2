# policy_iteration.py
"""Volume 2: Policy Function Iteration.
<Sam Goldrup>
<MATH 323>
<6 April 2022>
"""

import numpy as np

# Intialize P for test example
#Left =0
#Down = 1
#Right = 2
#Up= 3

P = {s : {a: [] for a in range(4)} for s in range(4)}
P[0][0] = [(0, 0, 0, False)]
P[0][1] = [(1, 2, -1, False)]
P[0][2] = [(1, 1, 0, False)]
P[0][3] = [(0, 0, 0, False)]
P[1][0] = [(1, 0, -1, False)]
P[1][1] = [(1, 3, 1, True)]
P[1][2] = [(0, 0, 0, False)]
P[1][3] = [(0, 0, 0, False)]
P[2][0] = [(0, 0, 0, False)]
P[2][1] = [(0, 0, 0, False)]
P[2][2] = [(1, 3, 1, True)]
P[2][3] = [(1, 0, 0, False)]
P[3][0] = [(0, 0, 0, True)]
P[3][1] = [(0, 0, 0, True)]
P[3][2] = [(0, 0, 0, True)]
P[3][3] = [(0, 0, 0, True)]



# Problem 1
def value_iteration(P, nS ,nA, beta = 1, tol=1e-8, maxiter=3000):
    """Perform Value Iteration according to the Bellman optimality principle.

    Parameters:
        P (dict): The Markov relationship
                (P[state][action] = [(prob, nextstate, reward, is_terminal)...]).
        nS (int): The number of states.
        nA (int): The number of actions.
        beta (float): The discount rate (between 0 and 1).
        tol (float): The stopping criteria for the value iteration.
        maxiter (int): The maximum number of iterations.

    Returns:
       v (ndarray): The discrete values for the true value function.
       n (int): number of iterations
    """
    V_old = np.zeros(nS) #old value function
    V_new = np.copy(V_old) #new value function
    n_iters = maxiter #track the number of iterations
    for k in range(maxiter):
        for s in range(nS):
            sa_vector = np.zeros(nA) #state action vector
            for a in range(nA):
                for tuple_info in P[s][a]:
                    p,s_,u,_ = tuple_info
                    sa_vector[a] += (p * (u + beta * V_old[s_])) #update each action portion of the vector
            V_new[s] = np.max(sa_vector) #get maximal element
        if np.linalg.norm(V_old-V_new) < tol: #convergence check!
            n_iters = k + 1
            break
        V_old = V_new.copy() #update old value function
    return V_new, n_iters

# Problem 2
def extract_policy(P, nS, nA, v, beta = 1.0):
    """Returns the optimal policy vector for value function v

    Parameters:
        P (dict): The Markov relationship
                (P[state][action] = [(prob, nextstate, reward, is_terminal)...]).
        nS (int): The number of states.
        nA (int): The number of actions.
        v (ndarray): The value function values.
        beta (float): The discount rate (between 0 and 1).

    Returns:
        policy (ndarray): which direction to move in from each square.
    """
    #iterate over all the actions in the state space then all the states
    pol = np.array([np.argmax(np.array([P[state][act][0][0] * (P[state][act][0][2]+beta*v[act]) for act in range(nA)])) for state in range(nS)])
        
    return pol

# Problem 3
def compute_policy_v(P, nS, nA, policy, beta=1.0, tol=1e-8):
    """Computes the value function for a policy using policy evaluation.

    Parameters:
        P (dict): The Markov relationship
                (P[state][action] = [(prob, nextstate, reward, is_terminal)...]).
        nS (int): The number of states.
        nA (int): The number of actions.
        policy (ndarray): The policy to estimate the value function.
        beta (float): The discount rate (between 0 and 1).
        tol (float): The stopping criteria for the value iteration.

    Returns:
        v (ndarray): The discrete values for the true value function.
    """
    pi_new = np.zeros(nS) #old policy vector
    pi_old = policy #new policy vector
    while True:
        for s in range(nS):
            sa_vector = np.zeros(nA) #state action vector
            for a in range(nA):
                for tuple_info in P[s][a]:
                    p,s_,u,_ = tuple_info
                    sa_vector[a] += (p * (u + beta * pi_old[s_])) #update state action vector
            pi_new[s] = np.max(sa_vector) #update state of policy vector
        if np.linalg.norm(pi_new-pi_old) < tol: #convergence check!
            break
        pi_old = pi_new.copy() #update old policy
    return pi_new

# Problem 4
def policy_iteration(P, nS, nA, beta=1, tol=1e-8, maxiter=200):
    """Perform Policy Iteration according to the Bellman optimality principle.

    Parameters:
        P (dict): The Markov relationship
                (P[state][action] = [(prob, nextstate, reward, is_terminal)...]).
        nS (int): The number of states.
        nA (int): The number of actions.
        beta (float): The discount rate (between 0 and 1).
        tol (float): The stopping criteria for the value iteration.
        maxiter (int): The maximum number of iterations.

    Returns:
    	v (ndarray): The discrete values for the true value function
        policy (ndarray): which direction to move in each square.
        n (int): number of iterations
    """
    pi_0 = np.random.choice(nA,size=nS) #old policy vector
    #sa_vector = np.zeros(nS)
    n_iters = maxiter
    for k in range(maxiter):
        v_new = compute_policy_v(P,nS,nA,pi_0,beta) #new value function
        pi_new = extract_policy(P,nS,nA,v_new,beta) #update policy based on new value function
        if np.linalg.norm(pi_new-pi_0) < tol: #convergence check
            n_iters = k #the number of iterations it took
            break
        pi_0 = pi_new.copy() #reset policy
    return v_new,pi_new,n_iters 

# Problem 5 and 6
def frozen_lake(basic_case=True, M=1000, render=False):
    """ Finds the optimal policy to solve the FrozenLake problem

    Parameters:
    basic_case (boolean): True for 4x4 and False for 8x8 environemtns.
    M (int): The number of times to run the simulation using problem 6.
    render (boolean): Whether to draw the environment.

    Returns:
    vi_policy (ndarray): The optimal policy for value iteration.
    vi_total_rewards (float): The mean expected value for following the value iteration optimal policy.
    pi_value_func (ndarray): The maximum value function for the optimal policy from policy iteration.
    pi_policy (ndarray): The optimal policy for policy iteration.
    pi_total_rewards (float): The mean expected value for following the policy iteration optimal policy.
    """
    raise NotImplementedError("Problem 5 Incomplete")

# Problem 6
def run_simulation(env, policy, render=True, beta = 1.0):
    """ Evaluates policy by using it to run a simulation and calculate the reward.

    Parameters:
    env (gym environment): The gym environment.
    policy (ndarray): The policy used to simulate.
    beta float: The discount factor.
    render (boolean): Whether to draw the environment.

    Returns:
    total reward (float): Value of the total reward received under policy.
    """
    raise NotImplementedError("Problem 6 Incomplete")

if __name__ == "__main__":
    # v = value_iteration(P,4,4)
    # print(v)
    # pol = extract_policy(P,4,4,v[0])
    # print(pol)
    # print(compute_policy_v(P,4,4,pol))
    # print(policy_iteration(P,4,4,1))
    pass