# dynamic_programming.py
"""Volume 2: Dynamic Programming.
<Samuel Goldrup>
<MATH 323>
<30 March 2022>
"""

import numpy as np
from matplotlib import pyplot as plt


def calc_stopping(N):
    """Calculate the optimal stopping time and expected value for the
    marriage problem.

    Parameters:
        N (int): The number of candidates.

    Returns:
        (float): The maximum expected value of choosing the best candidate.
        (int): The index of the maximum expected value.
    """
    values = [0] #value starts at 0
    for i in range(N-1,0,-1):
        values.append(max(values[N-i-1],(1/N)+(i/(i+1))*values[N-i-1])) #back solve
    return np.max(values),np.abs(N-np.argmax(values)) #optimizer and optimal value



# Problem 2
def graph_stopping_times(M):
    """Graph the optimal stopping percentage of candidates to interview and
    the maximum probability against M.

    Parameters:
        M (int): The maximum number of candidates.

    Returns:
        (float): The optimal stopping percent of candidates for M.
    """
    values = [] #to plot these lists
    stop_pct = []
    for N in range(3,M+1):
        values.append(calc_stopping(N)[0]) #optimizers
        stop_pct.append(calc_stopping(N)[1]/N) #/N to calculate percentage
    plt.plot(list(range(3,M+1)),values,label="value")
    plt.plot(list(range(3,M+1)),stop_pct,label="'%' of candidates")
    plt.xlabel("Candidate Pool Size")
    plt.title("Graph of Value and Optimal Stopping '%' against time")
    plt.legend() #get the labels
    plt.show()

    return stop_pct[-1]


# Problem 3
def get_consumption(N, u=lambda x: np.sqrt(x)):
    """Create the consumption matrix for the given parameters.

    Parameters:
        N (int): Number of pieces given, where each piece of cake is the
            same size.
        u (function): Utility function.

    Returns:
        C ((N+1,N+1) ndarray): The consumption matrix.
    """
    w = [i/N for i in range(N+1)] #cake ratios, consumption matrix
    C = np.array([[u(w[i]-w[j]) if w[i]-w[j] >= 0 else u(0) for j in range(N+1)] for i in range(N+1)])
    return C


# Problems 4-6
def eat_cake(T, N, B, u=lambda x: np.sqrt(x)):
    """Create the value and policy matrices for the given parameters.

    Parameters:
        T (int): Time at which to end (T+1 intervals).
        N (int): Number of pieces given, where each piece of cake is the
            same size.
        B (float): Discount factor, where 0 < B < 1.
        u (function): Utility function.

    Returns:
        A ((N+1,T+1) ndarray): The matrix where the (ij)th entry is the
            value of having w_i cake at time j.
        P ((N+1,T+1) ndarray): The matrix where the (ij)th entry is the
            number of pieces to consume given i pieces at time j.
    """
    #problem 4
    A, P = np.zeros((N+1,T+1)), np.zeros((N+1,T+1)) #initalize A and P
    w = np.linspace(0,1,N+1) #cake ratios (assume amount 1)
    A[:,-1] = u(w) #last column
    P[:,-1] = w #last column
    C = get_consumption(N,u) #get your consumption matrix
    #problem 5,6
    for t in range(T-1,-1,-1):
        CV = np.zeros((N+1,N+1)) #CV matrix for time t
        for i in range(N+1):
            for j in range(N+1):
                CV[i,j] = C[i,j] + B*A[j,(t+1)] #build CV
        CV_t = np.tril(CV,0) #CV_t
        for i in range(N+1):
            A[i,t] = max(CV_t[i]) #find largest vaule in each row and fill it into A
            j = np.argmax([CV_t[i]]) #the index used to built P[i,t]
            P[i,t] = w[i] - w[j]
        
    return A, P


# Problem 7
def find_policy(T, N, B, u=lambda x: np.sqrt(x)):
    """Find the most optimal route to take assuming that we start with all of
    the pieces. Show a graph of the optimal policy using graph_policy().

    Parameters:
        T (int): Time at which to end (T+1 intervals).
        N (int): Number of pieces given, where each piece of cake is the
            same size.
        B (float): Discount factor, where 0 < B < 1.
        u (function): Utility function.

    Returns:
        ((T,) ndarray): The matrix describing the optimal percentage to
            consume at each time.
    """
    A,P = eat_cake(T,N,B,u) #get the A and P
    opt_policy = [P[:,0][-1]] #first stage of policy
    cake = N

    for i in range(1,T+1):
        ate = int(np.round(opt_policy[-1]*N))
        cake -= ate #get new index for cake
        opt_policy.append(P[cake,i])

    return opt_policy

if __name__ == "__main__":
    #print(graph_stopping_times(1000))
    #print(get_consumption(4))
    # T,N,B = 3,4,0.9
    #print(eat_cake(T,N,B))
    # print(find_policy(T,N,B))
    # T,N,B = 4,5,0.9
    # u = lambda x: 3 - 3/(1+x)

    # print(find_policy(T,N,B,u))
    pass