# gradient_methods.py
"""Volume 2: Gradient Descent Methods.
Sam Goldrup
MATH 323
17 February 2022
"""

import numpy as np
import scipy.optimize as opt
from scipy import linalg as la
from autograd import numpy as anp
from autograd import grad
from matplotlib import pyplot as plt


# Problem 1
def steepest_descent(f, Df, x0, tol=1e-5, maxiter=100):
    """Compute the minimizer of f using the exact method of steepest descent.

    Parameters:
        f (function): The objective function. Accepts a NumPy array of shape
            (n,) and returns a float.
        Df (function): The first derivative of f. Accepts and returns a NumPy
            array of shape (n,).
        x0 ((n,) ndarray): The initial guess.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        ((n,) ndarray): The approximate minimum of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    num_iters = maxiter
    converged = False
    for i in range(maxiter):
        g = lambda alpha: f(x0-alpha*Df(x0).T) #what we wish to minimize (the descent w.r.t alpha)
        alpha =opt.minimize_scalar(g).x #the dot x makes it just return the optimal alpha

        x1 = x0 - alpha*Df(x0)
        
        if np.linalg.norm(Df(x0), ord=np.inf) <  tol: #convergence check
            num_iters = i + 1
            converged = True
            break
        x0 = x1
    return x0,converged,num_iters


# Problem 2
def conjugate_gradient(Q, b, x0, tol=1e-4):
    """Solve the linear system Qx = b with the conjugate gradient algorithm.

    Parameters:
        Q ((n,n) ndarray): A positive-definite square matrix.
        b ((n, ) ndarray): The right-hand side of the linear system.
        x0 ((n,) ndarray): An initial guess for the solution to Qx = b.
        tol (float): The convergence tolerance.

    Returns:
        ((n,) ndarray): The solution to the linear system Qx = b.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    r0 = Q@x0 - b #get initial r0,do
    d0 = -r0
    dk = d0 #prepare for the loop
    k = 0
    rk = r0
    n = len(b) #the method is designed to converge in less than n iters if Q is sparse enough
    converged = False
    while np.linalg.norm(rk) > tol and k < n:
        alph_k = (rk.T @ rk) / (dk.T @ Q @ dk) #get optimal alpha
        x1 = x0 + alph_k * dk #update position
        rk1 = rk + alph_k * Q @ dk #update rk
        beta = (rk1.T @ rk1) / (rk.T @ rk) #get a multiplier
        dk1 = -rk1 + beta*dk #update dk
        k += 1
        x0,rk,dk = x1,rk1,dk1 #update old values to new ones
    if np.linalg.norm(rk) < tol:
        converged = True
    return x1,converged,k


# Problem 3
def nonlinear_conjugate_gradient(f, df, x0, tol=1e-5, maxiter=100):
    """Compute the minimizer of f using the nonlinear conjugate gradient
    algorithm.

    Parameters:
        f (function): The objective function. Accepts a NumPy array of shape
            (n,) and returns a float.
        Df (function): The first derivative of f. Accepts and returns a NumPy
            array of shape (n,).
        x0 ((n,) ndarray): The initial guess.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        ((n,) ndarray): The approximate minimum of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    r0 = -df(x0).T 
    d0 = r0
    phi = lambda alpha: f(x0 + alpha*d0) #get function on which to argmin over alpha
    alph = opt.minimize_scalar(phi).x
    x1 = x0 + alph * d0 #first iteration
    xk,rk,dk = x1,r0,d0 #update and prepare for loop
    k = 1

    converged = False #default
    while np.linalg.norm(rk) >= tol and k < maxiter:
        rk1 = -df(xk).T
        beta = (rk1.T @ rk1) / (rk.T @ rk) #multiplier
        dk1 = rk1 + beta*dk
        g = lambda alpha: f(xk + alpha*dk1) #minimization method
        alph = opt.minimize_scalar(g).x
        xk1 = xk + alph * dk1 #update point
        k += 1
        xk,rk,dk = xk1,rk1,dk1 #prep for next iteration
    if np.linalg.norm(rk) < tol: #terminating condition
        converged = True
    
    return xk,converged,k



# Problem 4
def prob4(filename="linregression.txt",
          x0=np.array([-3482258, 15, 0, -2, -1, 0, 1829])):
    """Use conjugate_gradient() to solve the linear regression problem with
    the data from the given file, the given initial guess, and the default
    tolerance. Return the solution to the corresponding Normal Equations.
    """
    stata = np.loadtxt(filename)
    y_vals = stata[:,0]
    X = stata[:,1:]
    X = np.hstack((np.ones((len(y_vals),1)),X)) #append ones to left side of matrix

    soln = conjugate_gradient(X.T @ X,X.T @ y_vals,x0)[0] #solve the system
    return soln


# Problem 5
class LogisticRegression1D:
    """Binary logistic regression classifier for one-dimensional data."""

    def fit(self, x, y, guess):
        """Choose the optimal beta values by minimizing the negative log
        likelihood function, given data and outcome labels.

        Parameters:
            x ((n,) ndarray): An array of n predictor variables.
            y ((n,) ndarray): An array of n outcome variables.
            guess (array): Initial guess for beta.
        """
        m = len(x)
        f = lambda beta: anp.sum([anp.log(1+anp.exp(-(beta[0]+beta[1]*x[i]))) + (1-y[i])*(beta[0]+beta[1]*x[i]) for i in range(m)])
        #log-likelihood function
        df = grad(f) #derivative
        self.b0, self.b1 = opt.fmin_cg(f, anp.array(guess, dtype=anp.float64),fprime=df) #find minimizer


    def predict(self, x):
        """Calculate the probability of an unlabeled predictor variable
        having an outcome of 1.

        Parameters:
            x (float): a predictor variable with an unknown label.
        """
        return 1/(1+anp.exp(-(self.b0+self.b1*x))) #logit function


# Problem 6
def prob6(filename="challenger.npy", guess=np.array([20., -1.])):
    """Return the probability of O-ring damage at 31 degrees Farenheit.
    Additionally, plot the logistic curve through the challenger data
    on the interval [30, 100].

    Parameters:
        filename (str): The file to perform logistic regression on.
                        Defaults to "challenger.npy"
        guess (array): The initial guess for beta.
                        Defaults to [20., -1.]
    """
    data = np.load(filename)
    x,y = data[:,0], data[:,1] #temps and damage
    poop = LogisticRegression1D()
    poop.fit(x,y,guess) #get betas
    f_space = np.linspace(30,100,1000) #temps of interest
    plt.plot(f_space,[poop.predict(f) for f in f_space]) #plot predictions
    plt.scatter(x,y,label="Previous Damage",color="red")
    plt.scatter(31,poop.predict(31),label="P(Damage) at Launch",color="purple")
    plt.xlabel("Temperature")
    plt.ylabel("O-Ring Damage")
    plt.title("Probability of O-Ring Damage")
    plt.axis([30,100,-0.1,1.1]) #temps of ineterest
    plt.legend()
    plt.show()

    return poop.predict(31)

if __name__ == "__main__":
    # f = lambda x: x[0]**4 + x[1]**4 + x[2]**4
    # df = lambda x: np.array([4*x[0]**3,4*x[1]**3,4*x[2]**3])
    # x0 = np.array([np.pi, 2.7, -69])
    # print("simple quadratic:",steepest_descent(f, df, x0, tol=1e-5, maxiter=10000))


    # g = lambda x: (1-x[0])**2 + 100*(x[1]-x[0]**2)**2
    # dg = lambda x: np.array([-2*(1-x[0])-4*100*x[0]*(x[1]-x[0]**2), 2*100*(x[1]-x[0]**2)])
    # x0 = np.array([0,0])
    # print("rosen:",steepest_descent(g, dg, x0, tol=1e-5, maxiter=10000))

    # Q,b = np.array([[2,0],[0,4]]), np.array([1,8])
    # x0_list = [np.array([-100,100]), np.array([-200,-500]), np.array([420,-420]), np.array([690,420])]
    # for x0 in x0_list:
    #     print("conjugate (x0={}) :".format(x0), conjugate_gradient(Q, b, x0, tol=1e-4))

    # #makes randomn R^4 to R functions

    # n = 4
    # A = np.random.random((n,n))
    # Q = A.T @ A
    # b, x0 = np.random.random((2,n))

    # x = la.solve(Q,b)
    # my_x,convergence,num_iters = conjugate_gradient(Q,b,x0)
    # print("problem 2 test:",np.allclose(Q@my_x,b))
    # print("convergence:", convergence)

    # f = opt.rosen
    # df = opt.rosen_der
    # x0 = np.array([10,10])
    # print(opt.fmin_cg(f,x0,df))

    # print(nonlinear_conjugate_gradient(f, df, x0, tol=1e-5, maxiter=500))

    # print(prob4())

    # print(prob6())
    pass
