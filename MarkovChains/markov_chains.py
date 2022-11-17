# markov_chains.py
"""Volume 2: Markov Chains.
Sam Goldrup
Math 321
9 November 2021
"""

import numpy as np
from scipy import linalg as la


class MarkovChain:
    """A Markov chain with finitely many states.

    Attributes:
        (fill this out)
    """
    # Problem 1
    def __init__(self, A, states=None):
        """Check that A is column stochastic and construct a dictionary
        mapping a state's label to its index (the row / column of A that the
        state corresponds to). Save the transition matrix, the list of state
        labels, and the label-to-index dictionary as attributes.

        Parameters:
        A ((n,n) ndarray): the column-stochastic transition matrix for a
            Markov chain with n states.
        states (list(str)): a list of n labels corresponding to the n states.
            If not provided, the labels are the indices 0, 1, ..., n-1.

        Raises:
            ValueError: if A is not square or is not column stochastic.

        Example:
            >>> MarkovChain(np.array([[.5, .8], [.5, .2]], states=["A", "B"])
        corresponds to the Markov Chain with transition matrix
                                   from A  from B
                            to A [   .5      .8   ]
                            to B [   .5      .2   ]
        and the label-to-index dictionary is {"A":0, "B":1}.
        """
        self.A = A
        self.m,self.n = np.shape(A)
        self.states = states
        if not np.allclose(np.ones(self.n), A.sum(axis=0)):
            raise ValueError("not column stochastic")

        if self.states == None:
            self.Labels = list(range(self.m))
        else:
            self.Labels = self.states
        Indices = [i for i in range(self.m)]
        
        self.map_dict = dict(zip(self.Labels,Indices))
        


    # Problem 2
    def transition(self, state):
        """Transition to a new state by making a random draw from the outgoing
        probabilities of the state with the specified label.

        Parameters:
            state (str): the label for the current state.

        Returns:
            (str): the label of the state to transitioned to.
        """
        col_index = self.map_dict[state]
        result = np.random.multinomial(1,self.A[:,col_index])
        new_label_index = np.argmax(result)
        new_state = self.Labels[new_label_index]
        return new_state

    # Problem 3
    def walk(self, start, N):
        """Starting at the specified state, use the transition() method to
        transition from state to state N-1 times, recording the state label at
        each step.

        Parameters:
            start (str): The starting state label.

        Returns:
            (list(str)): A list of N state labels, including start.
        """
        walk = [start]
        state = start
        for i in range(N-1):
            state = self.transition(state)
            walk.append(state)

        return walk

    # Problem 3
    def path(self, start, stop):
        """Beginning at the start state, transition from state to state until
        arriving at the stop state, recording the state label at each step.

        Parameters:
            start (str): The starting state label.
            stop (str): The stopping state label.

        Returns:
            (list(str)): A list of state labels from start to stop.
        """
        path = [start]
        state = start
        while state != stop:
            state = self.transition(state)
            path.append(state)
        return path

    # Problem 4
    def steady_state(self, tol=1e-12, maxiter=40):
        """Compute the steady state of the transition matrix A.

        Parameters:
            tol (float): The convergence tolerance.
            maxiter (int): The maximum number of iterations to compute.

        Returns:
            ((n,) ndarray): The steady state distribution vector of A.

        Raises:
            ValueError: if there is no convergence within maxiter iterations.
        """
        curr = np.random.random(self.m)
        k = 1
        norm = 1
        while norm > tol:
            A_k = np.linalg.matrix_power(self.A,k)
            curr = A_k @ curr
            norm = la.norm(self.A @ curr - curr)
            k += 1
            if k > maxiter:
                raise ValueError("no convergence")
        return A_k[:,0]

class SentenceGenerator(MarkovChain):
    """A Markov-based simulator for natural language.

    Attributes:
        (fill this out)
    """
    # Problem 5
    def __init__(self, filename):
        """Read the specified file and build a transition matrix from its
        contents. You may assume that the file has one complete sentence
        written on each line.
        """

        with open(filename,'r') as myfile: #initialize a list of words
            words = myfile.read().split()

        with open(filename,'r') as myfile: #for training
            lines = myfile.readlines()

        word_dict = {k:i for i,k in enumerate(set(words))} #make a mapping dict
        word_dict["$tart"] = len(word_dict)
        word_dict["$top"] = len(word_dict)

        T = np.zeros((len(word_dict),len(word_dict))) #transition matrix
        T[len(word_dict)-1,len(word_dict)-1] = 1 #stop state must transition to itself

        for line in lines:
            word_list = line.split()
            word_list.insert(0,"$tart")
            word_list.append("$top")
            for k in range(len(word_list)-1):
                T[word_dict[word_list[k+1]]][word_dict[word_list[k]]] += 1
        
        T = T / (np.sum(T,axis=0,keepdims=True))

        super().__init__(T,list(word_dict.keys()))
        #self.states.append("$tart")
        #self.states.append("$top")
      

    # Problem 6
    def babble(self):
        """Create a random sentence using MarkovChain.path().

        Returns:
            (str): A sentence generated with the transition matrix, not
                including the labels for the $tart and $top states.

        Example:
            >>> yoda = SentenceGenerator("yoda.txt")
            >>> print(yoda.babble())
            The dark side of loss is a path as one with you.
        """
        start = self.states[len(self.states)-2]
        stop = self.states[len(self.states)-1]
        path = self.path(start, stop)
        path.pop(0)
        path.pop()
        sentence = " ".join(path)

        return sentence