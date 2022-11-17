# quassian_quadrature.py
"""Volume 2: Gaussian Quadrature.
San Goldrup
Math 323
20 January 2022
"""

import numpy as np
from scipy import linalg as la
from scipy import sparse
from scipy.stats import norm
from matplotlib import pyplot as plt
from scipy.integrate import quad

class GaussianQuadrature:
    """Class for integrating functions on arbitrary intervals using Gaussian
    quadrature with the Legendre polynomials or the Chebyshev polynomials.
    """
    # Problems 1 and 3
    def __init__(self, n, polytype="legendre"):
        """Calculate and store the n points and weights corresponding to the
        specified class of orthogonal polynomial (Problem 3). Also store the
        inverse weight function w(x)^{-1} = 1 / w(x).

        Parameters:
            n (int): Number of points and weights to use in the quadrature.
            polytype (string): The class of orthogonal polynomials to use in
                the quadrature. Must be either 'legendre' or 'chebyshev'.

        Raises:
            ValueError: if polytype is not 'legendre' or 'chebyshev'.
        """
        self.n = n #number of points and weights to use
        if polytype != "legendre" and polytype != "chebyshev":
            raise ValueError("not chebyshev or legendre family")
        self.label = polytype #label of which class of polynomials to use
        if self.label == "legendre":
            self.w = lambda x: 1 / 1 #1 divided by the weight function
        else: #chebyshev
            self.w = lambda x: np.sqrt(1-x**2)

        self.points, self.weights = self.points_weights(self.n)

    # Problem 2
    def points_weights(self, n):
        """Calculate the n points and weights for Gaussian quadrature.

        Parameters:
            n (int): The number of desired points and weights.

        Returns:
            points ((n,) ndarray): The sampling points for the quadrature.
            weights ((n,) ndarray): The weights corresponding to the points.

            test_pts = np.array([(-1/3)*np.sqrt(5+2*np.sqrt(10/7)),(-1/3)*np.sqrt(5
    ...: -2*np.sqrt(10/7)),0,(1/3)*np.sqrt(5-2*np.sqrt(10/7)),(1/3)*np.sqrt(5+2*
    ...: np.sqrt(10/7))])

            test_weights = np.array([(322-13*np.sqrt(70))/900,(322+13*np.sqrt(70))/900,128/225,(322+13*np.sqrt(70))/900,(322-13*np.sqrt(70))/900])
        """

        alphas = [0]*n

        if self.label == "legendre":
            betas = [np.sqrt((k**2)/(4*k**2 - 1)) for k in range(1,n+1)] #sqrt the betas to build the jacobi matrix
            scale_fact = 2

        else: #chebyshev
            betas = [1/2]*n #sqrt the betas to build the jacobi matrix
            betas[0] = np.sqrt(1/2)
            scale_fact = np.pi

        diagonals = [betas[:n-1], alphas, betas[:n-1]]
        offsets = [-1,0,1]
        J = sparse.diags(diagonals, offsets, shape=(n,n)).toarray()
        
        points, evecs = la.eig(J)
        evec_first_entries = evecs[0]

        points = points.real 
        orders = np.argsort(points)
        points = points[orders] #we need to sort the interpolating points from smallest to largest
        evec_first_entries = evec_first_entries[orders] #reorder for the eigen vectors as well

        v0_arr = np.array([evec for evec in evec_first_entries]).real
        weights = scale_fact*v0_arr**2 #maybe could speed this up using an np array

        return points, weights #only the first weight matches what is in the lab file...


    # Problem 3
    def basic(self, f):
        """Approximate the integral of a f on the interval [-1,1]."""
        g = f(self.points) * self.w(self.points) #g = lambda x : f(x) * self.w(x)
        inner_prod = self.weights @ g
        return inner_prod

    # Problem 4
    def integrate(self, f, a, b):
        """Approximate the integral of a function on the interval [a,b].

        Parameters:
            f (function): Callable function to integrate.
            a (float): Lower bound of integration.
            b (float): Upper bound of integration.

        Returns:
            (float): Approximate value of the integral.
        """
        h = lambda x: f(((b-a)/2)*x + ((b+a)/2)) #translate the domain [-1,1] to [a,b]
        return (b/2-a/2)*self.basic(h)

    # Problem 6.
    def integrate2d(self, f, a1, b1, a2, b2):
        """Approximate the integral of the two-dimensional function f on
        the interval [a1,b1]x[a2,b2].

        Parameters:
            f (function): A function to integrate that takes two parameters.
            a1 (float): Lower bound of integration in the x-dimension.
            b1 (float): Upper bound of integration in the x-dimension.
            a2 (float): Lower bound of integration in the y-dimension.
            b2 (float): Upper bound of integration in the y-dimension.

        Returns:
            (float): Approximate value of the integral.
        """
        h = lambda x,y: f(((b1-a1)/2)*x + (b1+a1)/2, ((b2-a2)/2)*y + (b2+a2)/2)
        g = lambda x,y: h(x,y) * self.w(x) * self.w(y)

        #double sum
        esti = np.sum([np.sum([self.weights[i]*self.weights[j]*g(self.points[i],self.points[j]) for j in range(self.n)]) for i in range(self.n)])

        #scale it
        return (b1-a1)*(b2-a2)*(1/4)*esti


# Problem 5
def prob5():
    """Use scipy.stats to calculate the "exact" value F of the integral of
    f(x) = (1/sqrt(2 pi))e^((-x^2)/2) from -3 to 2. Then repeat the following
    experiment for n = 5, 10, 15, ..., 50.
        1. Use the GaussianQuadrature class with the Legendre polynomials to
           approximate F using n points and weights. Calculate and record the
           error of the approximation.
        2. Use the GaussianQuadrature class with the Chebyshev polynomials to
           approximate F using n points and weights. Calculate and record the
           error of the approximation.
    Plot the errors against the number of points and weights n, using a log
    scale for the y-axis. Finally, plot a horizontal line showing the error of
    scipy.integrate.quad() (which doesn’t depend on n).
    """
    poop = norm.cdf(2) - norm.cdf(-3)

    sizes = list(range(5,55,5))

    f = lambda x: (1/np.sqrt(2*np.pi)) * np.exp(-(x**2)/2)

    leg_int_err = []
    cheb_int_err = []
    scipy_err = []

    for n in sizes:
        legendre_int = GaussianQuadrature(n) #legendre method
        leg_integral = legendre_int.integrate(f,-3,2) #integrate from -3 to 2

        chebyshev_int = GaussianQuadrature(n,"chebyshev") #chebyshev method
        cheb_integral = chebyshev_int.integrate(f,-3,2) #integrate from -3 to 2

        scipy_int = quad(f,-3,2)[0]

        leg_int_err.append(np.abs(leg_integral-poop))
        cheb_int_err.append(np.abs(cheb_integral-poop))
        scipy_err.append(np.abs(scipy_int-poop))

    plt.semilogy(sizes,leg_int_err,label="legendre method")
    plt.semilogy(sizes,cheb_int_err,label="chebyshev method")
    plt.semilogy(sizes,scipy_err,label="scipy library")
    plt.ylabel("error")
    plt.xlabel("number of points and weights")
    plt.legend()
    plt.show()
