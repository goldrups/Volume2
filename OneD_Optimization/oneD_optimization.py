# oneD_optimization.py
"""Volume 2: One-Dimensional Optimization.
Sam Goldrup
Math 323
23 January 2021
"""

from configparser import MAX_INTERPOLATION_DEPTH
import numpy as np
from scipy import optimize

from scipy.optimize import linesearch
from autograd import numpy as anp
from autograd import grad

# Problem 1
def golden_section(f, a, b, tol=1e-5, maxiter=15):
    """Use the golden section search to minimize the unimodal function f.

    Parameters:
        f (function): A unimodal, scalar-valued function on [a,b].
        a (float): Left bound of the domain.
        b (float): Right bound of the domain.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        (float): The approximate minimizer of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    num_iter = maxiter
    converged = False
    x0 = (a+b)/2.0 #set the initial minimizer approximation as the interval midpoint
    phi = (1.0+np.sqrt(5.0))/2.0
    for i in range(1,maxiter+1): #iterate only maxiter times at most
        c = (b-a)/phi
        a_ = b-c
        b_ = a+c
        if f(a_) <= f(b_): #get new boundaries for search interval
            b = b_
        else:
            a = a_
        x1 = (a+b)/2.0 #update the minimizer approximation
        if np.abs(x1-x0) < tol:
            converged = True
            num_iter = i
            break #stop iterating if the approximation change gets slow
        x0=x1
    return x1, converged, num_iter


# Problem 2
def newton1d(df, d2f, x0, tol=1e-5, maxiter=15):
    """Use Newton's method to minimize a function f:R->R.

    Parameters:
        df (function): The first derivative of f.
        d2f (function): The second derivative of f.
        x0 (float): An initial guess for the minimizer of f.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        (float): The approximate minimizer of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    converged = False #these two things never change unless np.abs(x1-x0) < tol evaluates to True
    num_iter = maxiter
    for k in range(maxiter):
        x1 = x0 - (df(x0) / d2f(x0))
        if np.abs(x1-x0) < tol:
            converged = True
            num_iter = k + 1 #because we start with k=0
            break
        x0 = x1

    return x1, converged, num_iter


# Problem 3
def secant1d(df, x0, x1, tol=1e-5, maxiter=15):
    """Use the secant method to minimize a function f:R->R.

    Parameters:
        df (function): The first derivative of f.
        x0 (float): An initial guess for the minimizer of f.
        x1 (float): Another guess for the minimizer of f.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        (float): The approximate minimizer of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    converged = False
    num_iter = maxiter
    for k in range(maxiter):
        df_x1 = df(x1)
        df_x0 = df(x0) #compute once for each k, but ultimately its twice
        x2 = (x0*df_x1 - x1*df_x0) / (df_x1 - df_x0)
        if np.abs(x2-x1) < tol:
            converged = True
            num_iter = k + 1 #because we started with k=0
            break
        x0 = x1
        x1 = x2

    return x2, converged, num_iter




# Problem 4
def backtracking(f, Df, x, p, alpha=1, rho=.9, c=1e-4):
    """Implement the backtracking line search to find a step size that
    satisfies the Armijo condition.

    Parameters:
        f (function): A function f:R^n->R.
        Df (function): The first derivative (gradient) of f.
        x (float): The current approximation to the minimizer.
        p (float): The current search direction.
        alpha (float): A large initial step length.
        rho (float): Parameter in (0, 1).
        c (float): Parameter in (0, 1).

    Returns:
        alpha (float): Optimal step size.
    """
    Dfp = Df(x) @ p #precompute values
    fx = f(x)
    while (f(x+alpha*p)) > fx + c*alpha*Dfp: #wolfe condition
        alpha = rho*alpha
    return alpha

if __name__ == "__main__":
    #f = lambda x: np.exp(x) - 4*x
    #print("sammy:",golden_section(f,0,3,maxiter=30))
    #print("scipy:",optimize.golden(f,brack=(0,3)))

    # f = lambda x: x**2 + np.sin(5*x)
    # df = lambda x: 2*x + 5*np.cos(5*x)
    # d2f = lambda x : 2 - 25*np.sin(5*x)

    # print("sammy:",newton1d(df,d2f,0,tol=1e-10,maxiter=500))
    # print("scipy:",optimize.newton(df,x0=0,fprime=d2f,tol=1e-10,maxiter=500, full_output=True))


    # f = lambda x: x**2 + np.sin(x) + np.sin(10*x)
    # df = lambda x: 2*x + np.cos(x) + 10*np.cos(10*x)
    # print("sammy:",secant1d(df,x0=0,x1=-1,tol=1e-5,maxiter=500))
    # print("scipy:",optimize.newton(df, x0=0, tol=1e-5, maxiter=500, full_output=True))

    # f = lambda x: x[0]**2 + x[1]**2 + x[2]**2
    # Df = lambda x: np.array([2*x[0], 2*x[1], 2*x[2]])

    # x = anp.array([150., .03, 40.])
    # p = anp.array([-.5, -100., -4.5])
    # phi = lambda alpha: f(x + alpha*p)
    # dphi = grad(phi)
    # alpha, _ = linesearch.scalar_search_armijo(phi, phi(0.), dphi(0.))

    # alpha_guess = backtracking(f,Df,x,p)

    # print(alpha)
    # print(alpha_guess)
    pass