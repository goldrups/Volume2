# polynomial_interpolation.py
"""Volume 2: Polynomial Interpolation.
Sam Goldrup
Math
30 December 2021
"""

import numpy as np
from scipy.interpolate import BarycentricInterpolator
from scipy import linalg as la
from matplotlib import pyplot as plt
from numpy.fft import fft


# Problems 1 and 2
def lagrange(xint, yint, points):
    """Find an interpolating polynomial of lowest degree through the points
    (xint, yint) using the Lagrange method and evaluate that polynomial at
    the specified points.

    Parameters:
        xint ((n,) ndarray): x values to be interpolated.
        yint ((n,) ndarray): y values to be interpolated.
        points((m,) ndarray): x values at which to evaluate the polynomial.

    Returns:
        ((m,) ndarray): The value of the polynomial at the specified points.
    """
    L = np.array([[np.product(point-np.delete(xint,j))/np.product(xint[j]-np.delete(xint,j)) for point in points] for j in range(len(xint))])
    #each L_j is a row, so there are n_rows, each of length m
    yL = yint.reshape(-1,1)*L #multiply array of y_values by array of L_j's elementwise
    interp_poly = yL.sum(axis=0) #sum down the columns to get an array of m values
    return interp_poly


# Problems 3 and 4
class Barycentric:
    """Class for performing Barycentric Lagrange interpolation.

    Attributes:
        w ((n,) ndarray): Array of Barycentric weights.
        n (int): Number of interpolation points.
        x ((n,) ndarray): x values of interpolating points.
        y ((n,) ndarray): y values of interpolating points.
    """

    def __init__(self, xint, yint):
        """Calculate the Barycentric weights using initial interpolating points.

        Parameters:
            xint ((n,) ndarray): x values of interpolating points.
            yint ((n,) ndarray): y values of interpolating points.
        """
        self.x = xint
        self.y = yint
        self.n = len(self.x)
        self.w = np.ones(self.n) #initialize attributes

        C = (np.max(self.x) - np.min(self.x)) / 4 #get C factor
        shuffle = np.random.permutation(self.n-1) #to fight off numerical out of control ness

        for j in range(self.n):
            temp = (self.x[j] - np.delete(self.x, j)) / C #get diff and divide it all by C
            temp = temp[shuffle] #shuffle the elements around
            self.w[j] /= np.product(temp) #multiply them all and divide the jth weight by that product

    def __call__(self, points):
        """Using the calcuated Barycentric weights, evaluate the interpolating polynomial
        at points.

        Parameters:
            points ((m,) ndarray): Array of points at which to evaluate the polynomial.

        Returns:
            ((m,) ndarray): Array of values where the polynomial has been computed.
        """
        mask = [True if point in self.x else False for point in points] #this
        masky = [True if x in points else False for x in self.x] #this
        replacement_y = self.y[masky] #this
        top_arr = np.array([np.sum([((self.w[j]*self.y[j])/(point-self.x[j])) for j in range(len(self.x))]) for point in points])
        bot_arr = np.array([np.sum([((self.w[j])/(point-self.x[j])) for j in range(len(self.x))]) for point in points])
        bary_interp = top_arr / bot_arr
        bary_interp[mask] = replacement_y #this

        return bary_interp

    # Problem 4
    def add_weights(self, xint, yint):
        """Update the existing Barycentric weights using newly given interpolating points
        and create new weights equal to the number of new points.

        Parameters:
            xint ((m,) ndarray): x values of new interpolating points.
            yint ((m,) ndarray): y values of new interpolating points.
        """
        C_old = (np.max(self.x) - np.min(self.x)) / 4 #this is the old C factor

        C = (np.max(np.concatenate((self.x, xint))) - np.min(np.concatenate((self.x,xint)))) / 4 #new C

        xint = list(xint) #convert to lists
        yint = list(yint)

        self.x = list(self.x)
        self.y = list(self.y)
        self.w = list(self.w) 
        self.n = len(self.x) #n is the number of interpolating points


        while xint:
            new_y = yint.pop() #pop off the new y
            self.y.append(new_y) #add it on to self.y
            i = xint.pop() #pop off a new x pt
            #C_new = (np.max(self.x) - np.min(self.x)) / 4 #get the new C factor for the new set
            for j in range(len(self.w)): 
                self.w[j] /= (self.x[j] - i)
                self.w[j] *= (C**self.n)
                self.w[j] /= (C_old**(self.n-1))
            #self.w = self.w * (C**self.n)
            #self.w = self.w / (C_old**(self.n-1))
            new_weight = (C**self.n)/np.product([i-self.x[k] for k in range(len(self.w))]) #for a new weight we multiply it by C_new**n
            self.w.append(new_weight) #put that weight in
            self.x.append(i) #add that new point to x
            self.n = len(self.x) #update the length of the set
            C_old = C #now C_old=C for all iterations after the first one
                

        self.x = np.array(self.x)
        self.y = np.array(self.y)
        self.w = np.array(self.w)
        self.n = len(self.x)

        #OLD SHIT ALGORITHM
        #self.x = np.concatenate((self.x,xint)) #concatenate old and new
        #self.y = np.concatenate((self.y,yint)) 
        #self.n = len(self.x)
        #self.w = np.ones(self.n)
        
        # C = (np.max(self.x) - np.min(self.x)) / 4 #use constructor code
        # shuffle = np.random.permutation(self.n-1)

        # for j in range(self.n):
        #     temp = (self.x[j] - np.delete(self.x, j)) / C
        #     temp = temp[shuffle]
        #     self.w[j] /= np.product(temp)


# Problem 5
def prob5():
    """For n = 2^2, 2^3, ..., 2^8, calculate the error of intepolating Runge's
    function on [-1,1] with n points using SciPy's BarycentricInterpolator
    class, once with equally spaced points and once with the Chebyshev
    extremal points. Plot the absolute error of the interpolation with each
    method on a log-log plot.
    """
    dims = [2**k for k in list(range(2,9))] #2**2, 2**3, 2**4
    domain = np.linspace(-1,1,400) #make the domain
    f = lambda x: 1/(1+25 * x**2) #runge's function
    actual = f(domain) #real eyes realize real lies
    bary_errors = []
    cheby_bary_errors = []

    for dim in dims:
        pts = np.linspace(-1,1,dim)
        poly = BarycentricInterpolator(pts) #make a barycentric interpolator object
        poly.set_yi(f(pts))
        bary_eval = poly.__call__(domain) 
        bary_diff = bary_eval - actual #get diffs
        error_bary = la.norm(bary_diff,ord=np.inf) #use sup norm
        bary_errors.append(error_bary)
        
        cheby_pts = np.array([np.cos(j*np.pi/(dim)) for j in range(dim+1)]) #make another object using extremizers
        poly_cheby = BarycentricInterpolator(cheby_pts)
        poly_cheby.set_yi(f(cheby_pts))
        poly_cheby_eval = poly_cheby.__call__(domain)
        bary_cheby_diff = poly_cheby_eval - actual
        error_bary_cheby = la.norm(bary_cheby_diff, ord=np.inf)
        cheby_bary_errors.append(error_bary_cheby)

    plt.loglog(dims, bary_errors, base=2, label="equal spacing")
    plt.loglog(dims, cheby_bary_errors, base=2, label="chebyshev")
    plt.xlabel("number of interpolating pts")
    plt.ylabel("error")
    plt.title("erorr vs # of interp pts")
    plt.legend()
    plt.show()


# Problem 6
def chebyshev_coeffs(f, n):
    """Obtain the Chebyshev coefficients of a polynomial that interpolates
    the function f at n points.

    Parameters:
        f (function): Function to be interpolated.
        n (int): Number of points at which to interpolate.

    Returns:
        coeffs ((n+1,) ndarray): Chebyshev coefficients for the interpolating polynomial.
    """
    y = np.cos((np.pi * np.arange(2*n)) /n) #j=0,1,2,....,2n-1
    samples = f(y) #array broadcast
    coeffs = np.real(fft(samples))[:n+1] / n #use DFT
    coeffs[0] /= 2 #fill in numpy's failures
    coeffs[n] /= 2
    
    return coeffs


# Problem 7
def prob7(n):
    """Interpolate the air quality data found in airdata.npy using
    Barycentric Lagrange interpolation. Plot the original data and the
    interpolating polynomial.

    Parameters:
        n (int): Number of interpolating points to use.
    """ 
    airdata = np.load("airdata.npy") #loads data
    data_length = len(airdata) #gets length of array
    fx = lambda a, b, n: .5*(a+b + (b-a) * np.cos(np.arange(n+1) * np.pi / n)) #a function to get the chebyshev extremizers
    a, b = 0, 366 - 1/24 #the inputs
    domain = np.linspace(0,b,data_length) #a point for each hour
    points = fx(a,b,n) #gets those cheby exts
    temp = np.abs(points-domain.reshape(data_length,1)) #finds the difference between those chebbies and the points in "domain"
    temp2 = np.argmin(temp,axis=0) #finds which ones are minimized

    poly = BarycentricInterpolator(domain[temp2])
    poly.set_yi(airdata[temp2])
    eval = poly.__call__(domain)

    plt.scatter(np.array(list(range(data_length)))/24,airdata,s=0.1, label="data")
    plt.plot(domain,eval,lw=2,color='m',label="interpolation")
    plt.xlabel("Days in 2016")
    plt.ylabel("PM 2.5")
    plt.title("Interpolating Hourly PM 2.5 Data in Salt Lake County")
    plt.legend()
    plt.show()


    
