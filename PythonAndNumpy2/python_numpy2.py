# python_intro.py
"""Python Essentials: Introduction to Python.
Sam Goldrup
321 Section 3
7 September 2021
"""

import numpy as np

#Problem 1
def isolate(a, b, c, d, e): 
	print(a, b, c, sep='     ', end=" ") #print a,b,c five spaces apart
	print(d, e, sep = ' ', end = '\n') #on the same line, print d,e one space apart 

#Problem 2
def first_half(string):
	str_len = len(string) #get the length of the string
  
	half_index = int(str_len / 2) #an integer either less than or equal to half the length
	
	return string[:half_index] #gets all the letters before the half_index index


def backward(first_string):
	i = len(first_string) - 1 #we want to iterate from the end of the string
	reversed_str = "" 

	while i >= 0: 
		reversed_str += first_string[i] #append letters to reversed_str in our desired order
		i -= 1
  		
	return reversed_str

#Problem 3
def list_ops():
	animals = ["bear", "ant", "cat", "dog"] #define list
	animals.append("eagle") #add "eagle" to the end
	animals.remove("cat") #delete "cat" from it
	animals.insert(2, "fox") #put "fox" at the 2nd index
	animals.remove("ant") #delete "ant" 
	animals.sort(reverse=True) #reverse the order of the list 
	animals.remove("eagle") #delete "eagle"
	animals.insert(1, "hawk") #insert hawk at the 1st index
	animals[-1] += "hunter" #changes "bear" to "bearhunter"
	return animals

#Problem 4
def alt_harmonic(n):
	"""Return the partial sum of the first n terms of the alternating
	harmonic series. Use this function to approximate ln(2).
	"""
	sequence = [((-1)**(i+1))/i for i in range(1,n+1)]
	#each element in this list is ((-1)**(i+1))/i for i=1,2,...,n
	return sum(sequence)



def prob5(A):
	"""Make a copy of 'A' and set all negative entries of the copy to 0.
	Return the copy.

	Example:
		>>> A = np.array([-3,-1,3])
		>>> prob4(A)
		array([0, 0, 3])
	"""
	cop_arr = np.copy(A) #make a copy of the array
	mask = A < 0 #the mask is an array of Trues and Falses 
	cop_arr[mask] = 0 #turns the negative elements into zeros
	return cop_arr

def prob6():
	"""Define the matrices A, B, and C as arrays. Return the block matrix
                               | 0 A^T I |
                               | A  0  0 |,
                               | B  0  C |
	where I is the 3x3 identity matrix and each 0 is a matrix of all zeros
	of the appropriate size.
	"""
	
	A = np.arange(6).reshape(3,2)
	A = A.T #A matrix
	A_T = A.T #A_T matrix
	I = np.eye(3,3) #Identity
	B = np.full_like(I, 3) 
	B = np.tril(B) #B matrix
	C = np.diag([-2,-2,-2]) #C matrix
	z_1 = np.zeros((3,3)) #required zero matrices
	z_2 = np.zeros((2,2))
	z_3 = np.zeros((3,2))
	z_4 = np.zeros((2,3))
	
	chunk_1 = np.vstack((z_1,A,B)) #left 3 columns
	chunk_2 = np.vstack((A_T,z_2,z_3)) #middle two columns
	chunk_3 = np.vstack((I, z_4, C)) #right 3 columns

	return np.hstack((chunk_1, chunk_2, chunk_3)) 

def prob7(A):
	"""Divide each row of 'A' by the row sum and return the resulting array.

	Example:
		>>> A = np.array([[1,1,0],[0,1,0],[1,1,1]])
		>>> prob6(A)
		array([[ 0.5       ,  0.5       ,  0.        ],
				 [ 0.        ,  1.        ,  0.        ],
				 [ 0.33333333,  0.33333333,  0.33333333]])
	"""
	num_rows = np.shape(A)[0] #gets the number of rows
	row_sums = A.sum(axis=1) #adds values along the rows
	row_sums = row_sums.reshape((num_rows, 1)) #reshapes to a column_vector
	return A / row_sums



def prob8():
	grid = np.load("grid.npy")
	"""Given the array stored in grid.npy, return the greatest product of four
	adjacent numbers in the same direction (up, down, left, right, or
	diagonally) in the grid.
	"""
	#finds the maximum possible product of numbers in the...
	hori = np.max(grid[:,:-3] * grid[:,1:-2] * grid[:,2:-1] * grid[:,3:]) #horizontal direction
	vert = np.max(grid[:-3,:] * grid[1:-2,:] * grid[2:-1,:] * grid[3:,:]) #vertical direction
	diag_l = np.max(grid[:-3,:-3] * grid[1:-2,1:-2] * grid[2:-1,2:-1] * grid[3:, 3:]) #leftward diagonal direction
	diag_r = np.max(grid[:-3,3:] * grid[1:-2,2:-1] * grid[2:-1,1:-2] * grid[3:,:-3]) #rightward diagonal direction

	return max(hori, vert, diag_l, diag_r)



if __name__ == "__main__": #test my code
	isolate(1,2,3,4,5)
	print(first_half("helloguys"))
	print(backward("syugolleh"))
	print(list_ops())
	print(alt_harmonic(500))
	A = np.array([-3,-1,3])
	print(prob5(A))
	print(prob6())
	A = np.array([[1,1,0],[0,1,0],[1,1,1]])
	print(prob7(A))
	print(prob8())

