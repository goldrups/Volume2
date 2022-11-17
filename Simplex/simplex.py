"""Volume 2: Simplex

Sam Goldrup
23 February 2022
Math 323
"""

from multiprocessing.sharedctypes import Value
from re import X
import numpy as np


# Problems 1-6
class SimplexSolver(object):
    """Class for solving the standard linear optimization problem

                        minimize        c^Tx
                        subject to      Ax <= b
                                         x >= 0
    via the Simplex algorithm.
    """
    # Problem 1
    def __init__(self, c, A, b):
        """Check for feasibility and initialize the dictionary.

        Parameters:
            c ((n,) ndarray): The coefficients of the objective function.
            A ((m,n) ndarray): The constraint coefficients matrix.
            b ((m,) ndarray): The constraint vector.

        Raises:
            ValueError: if the given system is infeasible at the origin.
        """
        self.m,self.n = A.shape
        x = np.zeros(self.n) #use this to test feasibility at origin
        Ax = A @ x
        if (Ax<=b).all(): #feasibility requirement
            self.A = A
            self.b = b
            self.c = c
        else:
            raise ValueError("not feasible at origin")

        self._generatedictionary(self.c,self.A,self.b) #call function below

    # Problem 2
    def _generatedictionary(self, c, A, b):
        """Generate the initial dictionary.

        Parameters:
            c ((n,) ndarray): The coefficients of the objective function.
            A ((m,n) ndarray): The constraint coefficients matrix.
            b ((m,) ndarray): The constraint vector.
        """
        c_ = np.concatenate((c,np.zeros(self.m))).T #construct the matrix that represents the linear system
        A_ = np.hstack((A,np.eye(self.m)))
        Dright = np.vstack((c_,-A_)) #right half of system
        Dleft = np.concatenate((np.array([0]),b)) #left half of system
        Dleft = np.array([[d] for d in Dleft])
        self.D = np.hstack((Dleft,Dright))


    # Problem 3a
    def _pivot_col(self):
        """Return the column index of the next pivot column.
        """
        focus_row = self.D[0,1:] #look at equation, will return first var with negative coeff
        neg_indices = np.where(focus_row < 0)
        return neg_indices[0][0]+1 #add 1 because we use n-1 of the entries

    # Problem 3b
    def _pivot_row(self, index):
        """Determine the row index of the next pivot row using the ratio test
        (Bland's Rule).
        """    
        col_index = index
        left_col, focus_col = self.D[1:,0], self.D[1:,col_index]
        if np.mean(focus_col >= 0) == 1: #check for boundedness of problem
            raise ValueError("problem unbounded bro!")


        ratios = np.array([-left_col[i]/focus_col[i] if focus_col[i] <0 else np.inf for i in range(len(focus_col))]) #only calculate negs

        for i in range(len(ratios)):
            if ratios[i] > 0:
                if i == np.argmin(ratios): #if it is the min
                    return i+1
                else:
                    continue
            else:
                ratios[i] = np.max(np.abs(ratios)) + 1 #useless now


    # Problem 4
    def pivot(self):
        """Select the column and row to pivot on. Reduce the column to a
        negative elementary vector.
        """
        piv_col = self._pivot_col() #get the pivot column
        piv_row = self._pivot_row(piv_col) #use pivot column to get pivot row
        self.D[piv_row] /= (-1)*self.D[piv_row,piv_col] #stochasticize
        for i in range(len(self.D)): #row reduce
            if i != piv_row:
                self.D[i] += self.D[i,piv_col]*self.D[piv_row]
            else:
                continue

    # Problem 5
    def solve(self):
        """Solve the linear optimization problem.

        Returns:
            (float) The minimum value of the objective function.
            (dict): The basic variables and their values.
            (dict): The nonbasic variables and their values.
        """
        while np.mean(self.D[0,1:] < 0) > 0: #pivot until there are no more negative coeffs in objective function
            self.pivot() 
        min = self.D[0,0]
        finals = self.D[0,1:] #final f
        deps = {} #dictionaries of dependent and independent variables
        indeps = {}
        for i in range(len(finals)):
            if finals[i] == 0: #this condition sorts dependent variables
                deps[i] = self.D[np.argmin(self.D[:,i+1])][0]
            else:
                indeps[i] = 0
        return (min,deps,indeps)
            

# Problem 6
def prob6(filename='productMix.npz'):
    """Solve the product mix problem for the data in 'productMix.npz'.

    Parameters:
        filename (str): the path to the data file.

    Returns:
        ((n,) ndarray): the number of units that should be produced for each product.
    """
    data = np.load(filename)
    a_mat,prices,resources,demand = data['A'], data['p'], data['m'], data['d']
    c = -1*prices
    n = len(a_mat[0])
    I_n = np.eye(n)
    b = np.concatenate((resources,demand)) #full constraints
    A = np.vstack((a_mat,I_n)) #what we subject the constraints to
    market = SimplexSolver(c,A,b) #build the object
    min,indep,dep = market.solve() #report quantities

    quants = []
    
    for i in range(len(c)):
        if i in indep.keys():
            quants.append(indep[i])
        elif i in dep.keys():
            quants.append(dep[i])

    return np.array(quants)

if __name__ == "__main__":
    pass