# exceptions_fileIO.py
"""Python Essentials: Exceptions and File Input/Output.
Sam Goldrup
MATH 321
22 September 2021
"""

from random import choice


# Problem 1
def arithmagic():
    """
    Takes in user input to perform a magic trick and prints the result.
    Verifies the user's input at each step and raises a
    ValueError with an informative error message if any of the following occur:

    The first number step_1 is not a 3-digit number.
    The first number's first and last digits differ by less than $2$.
    The second number step_2 is not the reverse of the first number.
    The third number step_3 is not the positive difference of the first two numbers.
    The fourth number step_4 is not the reverse of the third number.
    """

    step_1 = input("Enter a 3-digit number where the first and last digits differ by 2 or more: ")
    if len(step_1) != 3:
    	raise ValueError("not a 3-digit number")
    if abs(int(step_1[0]) - int(step_1[-1])) < 2:
    	raise ValueError("first and last digit differ by less than 2")
    step_2 = input("Enter the reverse of the first number, obtained by reading it backwards: ")
    if step_2[len(step_2)::-1] != step_1:
    	raise ValueError("This is not the reverse")
    step_3 = input("Enter the positive difference of these numbers: ")
    if int(step_3) != abs(int(step_2) - int(step_1)):
    	raise ValueError("This is not the positive difference")
    step_4 = input("Enter the reverse of the previous result: ")
    if step_4[len(step_4)::-1] != step_3:
    	raise ValueError("This is not the reverse")
    diff_reverse_sum = int(step_3) + int(step_4)
    print(str(step_3), "+", str(step_4), "=", str(diff_reverse_sum), "(ta-da!)")


# Problem 2
def random_walk(max_iters=1e12):
    """
    If the user raises a KeyboardInterrupt by pressing ctrl+c while the
    program is running, the function should catch the exception and
    print "Process interrupted at iteration $i$".
    If no KeyboardInterrupt is raised, print "Process completed".

    Return walk.
    """

    walk = 0
    directions = [1, -1]
    try:
    	for i in range(int(max_iters)):
    		walk += choice(directions)
    except KeyboardInterrupt: #stops the random walk if user types Ctrl+C
    	print("process interrupted at iteration", i)
    else:
    	print("process completed")

    return walk


# Problems 3 and 4: Write a 'ContentFilter' class.
    """Class for reading in file

    Attributes:
        filename (str): The name of the file
        contents (str): the contents of the file

    """
class ContentFilter(object):
    # Problem 3
    def __init__(self, filename):
    	"""Read from the specified file. If the filename is invalid, prompt
    	the user until a valid filename is given.
    	"""
    	run = True
    	while run == True: #loop runs until valid filename is entered
    		try:
    			myfile = open(filename, "r")
    			run = False
    		except (FileNotFoundError, TypeError, OSError):
    			filename = input("Please enter a valid filename:")
    			run = True

    	self.filename = filename
    	self.contents = myfile.read()
    	self.countchars = len(self.contents) #counts characters
    	self.countalpha = sum([a.isalpha() for a in self.contents]) #counts alpha characters
    	self.countnums = sum([a.isdigit() for a in self.contents]) #counts numerical characters
    	self.countwhite = sum([a.isspace() for a in self.contents]) #counts whitespace characters
    	self.countlines = len(self.contents.split("\n")) #counts number of lines
    	myfile.close()



 # Problem 4 ---------------------------------------------------------------
    def check_mode(self, mode = 'w'):
    	"""Raise a ValueError if the mode is invalid."""
    	possible_modes = ['w', 'x', 'a']
    	if mode not in possible_modes:
    		raise ValueError("not a valid mode")

    def uniform(self, outfile, mode='w', case='upper'):
    	"""Write the data to the outfile in uniform case."""
    	self.check_mode(mode)
    	possible_cases = ["lower","upper"]
    	lines = self.contents
    	if case not in possible_cases: #raises ValueError for invalid case
    		raise ValueError("not a valid case")
    	lines = "".join(lines)
    	if case == "upper":
    		lines = lines.upper()
    	if case == "lower":
    		lines = lines.lower()
    	with open(outfile, mode) as myfile: #refer to it as myfile
    			myfile.write(lines)

    def reverse(self, outfile, mode='w', unit='line'):
    	"""Write the data to the outfile in reverse order."""
    	self.check_mode(mode)
    	possible_units = ["word", "line"]
    	if unit not in possible_units:
    		raise ValueError("not a valid unit")
    	lines = self.contents
    	lines = lines.strip()
    	lines = lines.split('\n') #splits lines into elements of a list
    	if unit == "line":
    		lines = lines[::-1] #reverse order of lines
    		for i in range(len(lines)):
    			lines[i] += '\n'
    	if unit == "word":
    		for i in range(len(lines)):
    			lines[i] = lines[i].split(' ')
    			lines[i] = lines[i][::-1] #reverse order of words in each line
    			lines[i] = " ".join(lines[i]) #turns lines[i] from list to string
    			lines[i] += '\n'
    	with open(outfile, mode) as myfile:
    		myfile.writelines(lines)


    def transpose(self, outfile, mode='w'):
        """Write the transposed version of the data to the outfile."""
        self.check_mode(mode)
        lines = self.contents
        lines = lines.strip()
        lines = lines.split('\n') #splits lines into elements of a list
        for j in range(len(lines)):
        		lines[j] = lines[j].split(' ')
        new_lines = [] #list of lines
        for i in range(len(lines[0])):
        		new_line = []
        		for j in range(len(lines)):
        			new_let = lines[j][i] #new_letter to be appended to new_line
        			new_line.append(new_let)
        		new_line = " ".join(new_line) #turns new_line from list to string
        		new_line = new_line + "\n"
        		new_lines.append(new_line)
        with open(outfile, mode) as myfile:
        		myfile.writelines(new_lines)



        for row,d in zip(strings, vals):
        		output_string += "{:<25}{:<10} \n".format(row,d) #formats output in matrix style
        return output_string.strip()
    def __str__(self):
        """String representation: info about the contents of the file."""
        strings = ["Sourcefile:", "Total characters:", "Alphabetic characters:",
        "Numerical characters:", "Whitespace characters:", "Number of lines:"]
        vals = [self.filename, self.countchars, self.countalpha, self.countnums,
        self.countwhite, self.countlines]
        output_string = ""
        for row,d in zip(strings, vals):
        		output_string += "{:<25}{:<10} \n".format(row,d) #formats output in matrix style
        return output_string.strip()
