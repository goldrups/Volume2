# object_oriented.py
"""Python Essentials: Object Oriented Programming.
Sam Goldrup
Math 321
9 Sep 2021
"""
import math

class Backpack:
	"""A Backpack object class. Has a name, color, max_size and a list of contents.

	Attributes:
			name (str): the name of the backpack's owner.
			color (str): the color of the backpack
			max_size (int): the maximum number of items for the backpack
			contents (list): the contents of the backpack.
	"""

	# Problem 1: Modify __init__() and put(), and write dump().
	def __init__(self, name, color, max_size = 5):
		"""Set the name, color, and max_size.
		Initialize an empty list of contents.

		Parameters:
			name (str): the name of the backpack's owner.
			color (str): the color of the backpack
			max_size (int): maximum number of items for backpack (default 5)
		"""
		self.name = name
		self.color = color
		self.max_size = max_size
		self.contents = []


	def put(self, item):
		"""Check that backpack does not go over capacity.
		If the backpack is at full capacity already, the item is not added.
		The user is told there is no room in the backpack.
		"""
		if len(self.contents) < self.max_size:
			self.contents.append(item)
		else:
			print("No room!")

	def dump(self):
		"""
		Resets the contents of the backpack to an empty list
		"""
		self.contents = []

	def take(self, item):
		"""Remove an item from the backpack's list of contents."""
		self.contents.remove(item)

    # Magic Methods -----------------------------------------------------------

	 # Problem 3: Write __eq__() and __str__().
	def __add__(self, other):
		"""Add the number of contents of each Backpack."""
		return len(self.contents) + len(other.contents)

	def __lt__(self, other):
		"""Compare two backpacks. If 'self' has fewer contents
		than 'other', return True. Otherwise, return False.
		"""
		return len(self.contents) < len(other.contents)

	def __eq__(self, other):
		"""
		Compare the number of items in backpack, color and name.
		Create three bools, each of which evaluate to True if the attributes are equal
		return the three bools in a string of "and"s
		So the function returns True iff each bool evaluates to True
		"""
		contents_equal_length = len(self.contents) == len(other.contents)
		color_same = self.color == other.color
		name_same = self.name == other.name
		return contents_equal_length and color_same and name_same

	def __str__(self):
		"""
		Print the owner name, jetpack color, jetpack size,
		jetpack max size, and contents of jetpack.
		"""
		size_str = str(len(self.contents))
		max_size_str = str(self.max_size)

		a = "Owner:" + "\t\t" + self.name + '\n'
		b = "Color:" + '\t\t' + self.color + '\n'
		c = "Size:" + '\t\t' + size_str + '\n'
		d = "Max Size:" + '\t' + max_size_str + '\n'
		e = "Contents:" + '\t' + str(self.contents) + '\n'

		return a+b+c+d+e

def test_backpack():
	testpack = Backpack("Barry", "black") # Instantiate the object.
	if testpack.name != "Barry": # Test an attribute.
		print("Backpack.name assigned incorrectly")
	for item in ["pencil","pen","paper","computer"]:
		testpack.put(item)
	print("Contents:", testpack.contents)

# An example of inheritance. You are not required to modify this class.
class Knapsack(Backpack):
    """A Knapsack object class. Inherits from the Backpack class.
    A knapsack is smaller than a backpack and can be tied closed.

    Attributes:
        name (str): the name of the knapsack's owner.
        color (str): the color of the knapsack.
        max_size (int): the maximum number of items that can fit inside.
        contents (list): the contents of the backpack.
        closed (bool): whether or not the knapsack is tied shut.
    """
    def __init__(self, name, color):
        """Use the Backpack constructor to initialize the name, color,
        and max_size attributes. A knapsack only holds 3 item by default.

        Parameters:
            name (str): the name of the knapsack's owner.
            color (str): the color of the knapsack.
            max_size (int): the maximum number of items that can fit inside.
        """
        Backpack.__init__(self, name, color, max_size=3)
        self.closed = True

    def put(self, item):
        """If the knapsack is untied, use the Backpack.put() method."""
        if self.closed:
            print("I'm closed!")
        else:
            Backpack.put(self, item)

    def take(self, item):
        """If the knapsack is untied, use the Backpack.take() method."""
        if self.closed:
            print("I'm closed!")
        else:
            Backpack.take(self, item)

    def weight(self):
        """Calculate the weight of the knapsack by counting the length of the
        string representations of each item in the contents list.
        """
        return sum(len(str(item)) for item in self.contents)


# Problem 2: Write a 'Jetpack' class that inherits from the 'Backpack' class.

class Jetpack(Backpack):
	"""
	A Jetpack object class which inherits from the Backpack Object class
	Has name, color, max_size, list of contents, and amount of fuel.
	Attributes:
		name (str): name of the jetpack
		color (str): color of the jetpack
		max_size (int): maximum number of items the jetpack can hold
		contents (list): the items stored in the jetpack
		fuel (int): amount of fuel in the jetpack

	"""
	def __init__(self, name, color, max_size = 2, fuel=10):
		"""
		Use the Backpack constructor to initialize the name, color and max_size.
		New parameter 'fuel' is assigned to self.fuel
		"""
		Backpack.__init__(self, name, color, max_size)
		self.fuel = fuel

	def fly(self, fuel_to_burn):
		"""
		Accepts an input for the amount of fuel to be burned (int).
		If there is not enough fuel, the amount of fuel remains unchanged.
		Otherwise, it is decremented.
		"""
		if fuel_to_burn > self.fuel:
			print("Not enough fuel!")
		else:
			self.fuel -= fuel_to_burn

	def dump(self):
		"""
		Sets the list of contents to an empty list
		Assigns the self.fuel variable to 0
		"""
		Backpack.dump(self)
		self.fuel = 0


# Problem 4: Write a 'ComplexNumber' class.

class ComplexNumber:
	"""A ComplexNumber object class. Has a real and imaginary component

	Attributes:
			real (int): the a in a+bi
			imag (str): the b in a+bi
	"""
	def __init__(self, real, imag):
		"""
		Set the values of the real and imaginary component
		"""
		self.real = real
		self.imag = imag

	def conjugate(self):
		"""
		Create a new ComplexNumber object using -b instead
		"""
		my_complex_num = ComplexNumber(self.real, -self.imag)
		return my_complex_num

	def __str__(self):
		"""
		returns a string of the complex number
		"""
		str_real = str(self.real)
		str_imag = str(self.imag)
		if self.imag < 0: #if b < 0
			str_complex = "(" + str_real + str_imag + "j" + ")"
		if self.imag > 0: #if b > 0
			str_complex = "(" + str_real + "+" + str_imag + "j" + ")"
		return str_complex

	def __abs__(self):
		"""
		returns the absolute value of the complex number
		uses the formula sqrt(a^2 + b^2)
		"""
		a_sq = (self.real)**2
		b_sq = (self.imag)**2
		return math.sqrt(a_sq + b_sq)

	def __eq__(self, other):
		"""
		Checks a and b components of both values to see
		if the imaginary numbers are equal
		"""
		is_match = False
		if self.real == other.real and self.imag == other.imag:
			is_match = True
		return is_match

	def __add__(self, other):
		"""
		Adds the two a components and the two b components together
		creates a ComplexNumber object using those sums
		"""
		real_sum = self.real + other.real
		imag_sum = self.imag + other.imag
		complex_sum = ComplexNumber(real_sum, imag_sum)
		return complex_sum

	def __sub__(self, other):
		"""
		Take the difference of the a components and then the b components
		Create a new ComplexNumber object using those differences
		"""
		real_diff = self.real - other.real
		imag_diff = self.imag - other.imag
		complex_diff = ComplexNumber(real_diff, imag_diff)
		return complex_diff

	def __mul__(self, other):
		"""
		Use rules of multiplication to create the new a and b components
		Use those to create a new ComplexNumber object
		"""
		real_prod = (self.real * other.real) - (self.imag * other.imag)
		imag_prod = (self.real * other.imag) + (self.imag * other.real)
		complex_prod = ComplexNumber(real_prod, imag_prod)
		return complex_prod

	def __truediv__(self, other):
		"""
		Use rules of division to create the new a and b components
		Use those to create a new ComplexNumber object
		"""
		real_quot = (self.real * other.real) + (self.imag * other.imag)
		real_quot /= (((other.real)**2) + ((other.imag)**2))
		imag_quot = (self.imag * other.real) - (self.real * other.imag)
		imag_quot /= (((other.real)**2) + ((other.imag)**2))
		complex_quot = ComplexNumber(real_quot, imag_quot)
		return complex_quot

def test_ComplexNumber(a, b, c, d):
	"""
	Test different methods in the ComplexNumber class using 4
	different complex numbers
	"""
	py_cnum, my_cnum = complex(a, b), ComplexNumber(a, b)
	py_cnum_other, my_cnum_other = complex(c, d), ComplexNumber(c, d)

	# Validate the constructor.
	if my_cnum.real != a or my_cnum.imag != b:
		print("__init__() set self.real and self.imag incorrectly")

	# Validate conjugate() by checking the new number's imag attribute.
	if py_cnum.conjugate().imag != my_cnum.conjugate().imag:
		print("conjugate() failed for", py_cnum)

	# Validate __str__().
	if str(py_cnum) != str(my_cnum):
		print(str(py_cnum))
		print(str(my_cnum))
		print("__str__() failed for", py_cnum)

	# Validate __abs__().
	if abs(py_cnum) != abs(my_cnum):
		print("__abs__() failed for", py_cnum)

	# Validate __eq__().
	if py_cnum.__eq__(py_cnum_other) != my_cnum.__eq__(my_cnum_other):
		print("__eq__() failed for", py_cnum, "and", py_cnum_other)

	# Validate __add__().
	if py_cnum.__add__(py_cnum_other) != my_cnum.__add__(my_cnum_other):
		print("__add__() failed for", py_cnum, "and", py_cnum_other)

	# Validate __sub__().
	if py_cnum.__sub__(py_cnum_other) != my_cnum.__sub__(my_cnum_other):
		print("__sub__() failed for", py_cnum, "and", py_cnum_other)

	# Validate __mul__().
	if py_cnum.__mul__(py_cnum_other) != my_cnum.__mul__(my_cnum_other):
		print("__mul__() failed for", py_cnum, "and", py_cnum_other)

	# Validate __truediv__().
	if py_cnum.__truediv__(py_cnum_other) != my_cnum.__truediv__(my_cnum_other):
		print("__truediv__() failed for", py_cnum, "and", py_cnum_other)



#if __name__ == "__main__":
#	test_backpack()
#	test_ComplexNumber(3,4,5,6)
