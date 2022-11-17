# cvxpy_intro.py
"""Volume 2: Intro to CVXPY.
<Sam Goldrup>
<MATH 323>
<10 March 2022>
"""

import cvxpy as cp
import numpy as np

def prob1():
    """Solve the following convex optimization problem:

    minimize        2x + y + 3z
    subject to      x  + 2y         <= 3
                         y   - 4z   <= 1
                    2x + 10y + 3z   >= 12
                    x               >= 0
                          y         >= 0
                                z   >= 0

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    x = cp.Variable(3,nonneg=True) #initialize variable with nonnegative constraints
    c = np.array([2,1,3]) #coeffs
    A = np.array([0,1,-4])
    B = np.array([2,10,3])
    C = np.eye(3)

    objective = cp.Minimize(c.T @ x)
    constraints = [A @ x <= 1, B @ x >= 12, C @ x >= 0]
    problem = cp.Problem(objective, constraints)
    optimal = problem.solve() #get soln before the optimizer
    return x.value, optimal


# Problem 2
def l1Min(A, b):
    """Calculate the solution to the optimization problem

        minimize    ||x||_1
        subject to  Ax = b

    Parameters:
        A ((m,n) ndarray)
        b ((m, ) ndarray)

    Returns:
        The optimizer x (ndarray)
        The optimal value (float)
    """
    x = cp.Variable(len(A[0])) #dimensions must match so use len(A[0])
    objective = cp.Minimize(cp.norm(x,1)) #don't use la.norm
    constraints = [A@x==b]
    problem = cp.Problem(objective, constraints)

    optimal = problem.solve()
    return x.value, optimal


# Problem 3
def prob3():
    """Solve the transportation problem by converting the last equality constraint
    into inequality constraints.

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    p = cp.Variable(6,nonneg=True)
    c = np.array([4,7,6,8,8,9])
    objective = cp.Minimize(c.T @ p)
    A = np.array([[1,0,1,0,1,0],[0,1,0,1,0,1]]) #build constraint for these centers
    B = np.array([[1,0,0,1,0,0],[0,1,0,0,1,0],[0,0,1,0,0,1]]) #build constraint for those centers
    constraints = [A@p==np.array([5,8]), B@p==np.array([7,2,4])]
    
    problem = cp.Problem(objective, constraints)

    optimal = problem.solve()
    return p.value, optimal


# Problem 4
def prob4():
    """Find the minimizer and minimum of

    g(x,y,z) = (3/2)x^2 + 2xy + xz + 2y^2 + 2yz + (3/2)z^2 + 3x + z

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    Q = np.array([[3,2,1],[2,4,2],[1,2,3]]) #build quadratic matrix
    r = np.array([3,0,1])
    x = cp.Variable(3)
    prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(x,Q) + r.T @ x)) #plug in Q and r this way
    soln = prob.solve()
    return x.value, soln


# Problem 5
def prob5(A, b):
    """Calculate the solution to the optimization problem
        minimize    ||Ax - b||_2
        subject to  ||x||_1 == 1
                    x >= 0
    Parameters:
        A ((m,n), ndarray)
        b ((m,), ndarray)
        
    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    x = cp.Variable(len(A[0]),nonneg=True)
    objective = cp.Minimize(cp.norm(A@x-b,2)) #min 2 norm of the Ax-b
    constraints = [cp.sum(x)==1] #disciplined
    prob = cp.Problem(objective, constraints)

    soln = prob.solve()
    return x.value, soln


# Problem 6
def prob6():
    """Solve the college student food problem. Read the data in the file 
    food.npy to create a convex optimization problem. The first column is 
    the price, second is the number of servings, and the rest contain
    nutritional information. Use cvxpy to find the minimizer and primal 
    objective.
    
    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    data = np.load("food.npy", allow_pickle=True) #load in the data file baby
    x = cp.Variable(18,nonneg=True)
    p = np.array(data[:,0])
    objective = cp.Minimize(p.T @ x) #min cost
    #scale up by serving amount
    data[:,2],data[:,3],data[:,4],data[:,5],data[:,6],data[:,7]=data[:,2]*data[:,1],data[:,3]*data[:,1],data[:,4]*data[:,1],data[:,5]*data[:,1],data[:,6]*data[:,1],data[:,7]*data[:,1]
    #get the nutrition vectors
    cals,fats,sugar,calc,fibe,brotein = data[:,2],data[:,3],data[:,4],data[:,5],data[:,6],data[:,7]
    constraints = [cals.T@x<=2000, fats.T@ x<=65, sugar.T@x <=50,calc.T@x>=1000,fibe.T@x>=25,brotein.T@x>=46] #USDA constraints
    prob = cp.Problem(objective,constraints)
    soln = prob.solve()
    return x.value, soln

if __name__ == "__main__":
    # print("prob1:",prob1())
    # A = np.array([[1,2,1,1],[0,3,-2,-1]])
    # b = np.array([7,4])
    # print("prob2:",l1Min(A,b))
    # print("prob3:",prob3())
    # print("prob4:",prob4())
    # print("prob5:",prob5(A,b))
    # print("prob6:",prob6())
    pass