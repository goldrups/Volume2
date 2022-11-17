# interior_point_linear.py
"""Volume 2: Interior Point for Linear Programs.
<Sam Goldrup>
<MATH 347>
<24 March 2022>
"""

from cgi import parse
import numpy as np
from scipy import linalg as la
from scipy.stats import linregress
from matplotlib import pyplot as plt


# Auxiliary Functions ---------------------------------------------------------
def starting_point(A, b, c):
    """Calculate an initial guess to the solution of the linear program
    min c^T x, Ax = b, x>=0.
    Reference: Nocedal and Wright, p. 410.
    """
    # Calculate x, lam, mu of minimal norm satisfying both
    # the primal and dual constraints.
    B = la.inv(A @ A.T)
    x = A.T @ B @ b
    lam = B @ A @ c
    mu = c - (A.T @ lam)

    # Perturb x and s so they are nonnegative.
    dx = max((-3./2)*x.min(), 0)
    dmu = max((-3./2)*mu.min(), 0)
    x += dx*np.ones_like(x)
    mu += dmu*np.ones_like(mu)

    # Perturb x and mu so they are not too small and not too dissimilar.
    dx = .5*(x*mu).sum()/mu.sum()
    dmu = .5*(x*mu).sum()/x.sum()
    x += dx*np.ones_like(x)
    mu += dmu*np.ones_like(mu)

    return x, lam, mu

# Use this linear program generator to test your interior point method.
def randomLP(j,k):
    """Generate a linear program min c^T x s.t. Ax = b, x>=0.
    First generate m feasible constraints, then add
    slack variables to convert it into the above form.
    Parameters:
        j (int >= k): number of desired constraints.
        k (int): dimension of space in which to optimize.
    Returns:
        A ((j, j+k) ndarray): Constraint matrix.
        b ((j,) ndarray): Constraint vector.
        c ((j+k,), ndarray): Objective function with j trailing 0s.
        x ((k,) ndarray): The first 'k' terms of the solution to the LP.
    """
    A = np.random.random((j,k))*20 - 10
    A[A[:,-1]<0] *= -1
    x = np.random.random(k)*10
    b = np.zeros(j)
    b[:k] = A[:k,:] @ x
    b[k:] = A[k:,:] @ x + np.random.random(j-k)*10
    c = np.zeros(j+k)
    c[:k] = A[:k,:].sum(axis=0)/k
    A = np.hstack((A, np.eye(j)))
    return A, b, -c, x


# Problems --------------------------------------------------------------------
def interiorPoint(A, b, c, niter=20, tol=1e-16, verbose=False):
    """Solve the linear program min c^T x, Ax = b, x>=0
    using an Interior Point method.

    Parameters:
        A ((m,n) ndarray): Equality constraint matrix with full row rank.
        b ((m, ) ndarray): Equality constraint vector.
        c ((n, ) ndarray): Linear objective function coefficients.
        niter (int > 0): The maximum number of iterations to execute.
        tol (float > 0): The convergence tolerance.

    Returns:
        x ((n, ) ndarray): The optimal point.
        val (float): The minimum value of the objective function.
    """
    m,n = A.shape #get dimensions of A, should match up to x,mu,lamb
    A,b,c = A.astype(float), b.astype(float), c.astype(float) #cast as floats
    F = lambda x,lamb,mu: np.concatenate((A.T@lamb+mu-c,A@x-b,np.diag(mu)@x)) #2n+m vector
    sigma = 0.1
    #DF matrix
    DF = lambda x,mu: np.vstack((np.hstack((np.zeros((n,n)),A.T,np.eye(n,n))),np.hstack((A,np.zeros((m,m)),np.zeros((m,n)))),np.hstack((np.diag(mu),np.zeros((n,m)),np.diag(x)))))
    #centering parameter vector
    ctr_prmtr = lambda x,mu: np.concatenate((np.zeros(n),np.zeros(m),(sigma)*(1/n)*(x@mu)*np.ones(n)))
    def delta(x,lamb,mu):
        lu,piv = la.lu_factor(DF(x,mu))
        #solve for change in x,lamb,mu
        deltas = la.lu_solve((lu,piv),-F(x,lamb,mu)+ctr_prmtr(x,mu))
        return deltas

    def step_len(x,lamb,mu,step):
        deltax = step[:n] #delta for x vars
        deltamu = step[-n:] #delta for mu vars
        if np.mean(deltax >= 0) == 1: #all nonneg deltas
            delta_max = 1
        else:
            mask = deltax < 0 #where deltas are nonneg
            delta_max = np.min(-1*x[mask]/deltax[mask]) #minimize on these
        if np.mean(deltamu >= 0) == 1: #same process for choosing alpha
            alpha_max = 1
        else:
            mask = deltamu < 0
            alpha_max = np.min(-1*mu[mask]/deltamu[mask])

        alpha = np.min([1,0.95*alpha_max]) #get the min value
        delta = np.min([1,0.95*delta_max])
        return alpha, delta

    x0,lamb0,mu0 = starting_point(A, b, c)
    for i in range(niter):
        step = delta(x0,lamb0,mu0) #get direction
        alph,delt = step_len(x0,lamb0,mu0,step) #get amount to go in that direction
        x1 = x0 + delt*step[:n]
        lamb1 = lamb0 + alph*step[n:n+m]
        mu1 = mu0 + alph*step[-n:]
        x0,lamb0,mu0 = x1,lamb1,mu1 #update vals
        nu = (1/n)*(x0@mu0) #the convergence checker
        if np.abs(nu) < tol:
            break

    return x0, c@x0

def leastAbsoluteDeviations(filename='simdata.txt'):
    """Generate and show the plot requested in the lab."""
    with open(filename) as file:
        simdata = file.readlines()
    x, y = [], []
    for row in simdata:
        r = row.strip('\n').split(' ')
        # # first column: response variables (yi)
        y.append(float(r[0]))
        # # second column: explanatory variables (xi)
        x.append(float(r[1]))

    y_ = np.array(y) 
    x_ = np.array(x)
    data = np.hstack((y_.reshape(-1,1),x_.reshape(-1,1))) #bulid the data matrix
    m = data.shape[0] #run the code from the lab file lol
    n = data.shape[1] - 1
    c = np.zeros(3*m + 2*(n+1))
    c[:m] = 1
    y = np.empty(2*m)
    y[::2] = -data[:,0]
    y[1::2] = data[:, 0]
    x = data[:,1:]

    A = np.ones((2*m,3*m + 2*(n+1)))
    A[::2, :m] = np.eye(m)
    A[1::2, :m] = np.eye(m)
    A[::2, m:m+n] = -x
    A[1::2, m:m+n] = x
    A[::2, m+n:m+2*n] = x
    A[1::2, m+n:m+2*n] = -x
    A[::2, m+2*n] = -1
    A[1::2, m+2*n+1] = -1
    A[:, m+2*n+2:] = -np.eye(2*m, 2*m)

    sol = interiorPoint(A, y, c, niter=10)[0] #run interior point solver that i built with goated comp coding skills
    beta = sol[m:m+n] - sol[m+n:m+2*n] #LAD
    b = sol[m+2*n] - sol[m+2*n+1]

    slope, intercept = linregress(data[:,1],data[:,0])[:2] #OLS
    domain = np.linspace(0,10,200)

    plt.plot(domain,b+beta*domain,label="LAD") #LAD for Least Absolute Deviations
    plt.scatter(x_,y_,color="red",label="data")
    plt.plot(domain,intercept+slope*domain,label="OLS") #OLS for ordinary least squares
    plt.legend()
    plt.title("line fits")
    plt.ylabel("y vals")
    plt.xlabel("x vals")
    plt.show()

if __name__ == "__main__":
    # j,k = 3,2
    # A,b,c,x= randomLP(j,k)
    # point,value = interiorPoint(A,b,c)
    # print(c)
    # print(value)
    # print(x)
    # print(point[:k])

    # leastAbsoluteDeviations(filename='simdata.txt')
    pass
    


