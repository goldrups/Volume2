# shell2.py
"""Volume 3: Unix Shell 2.
Sam Goldrup
Math 321
November 11 2021
"""
import os
from glob import glob
import subprocess
import numpy as np


# Problem 3
def grep(target_string, file_pattern):
    """Find all files in the current directory or its subdirectories that
    match the file pattern, then determine which ones contain the target
    string.

    Parameters:
        target_string (str): A string to search for in the files whose names
            match the file_pattern.
        file_pattern (str): Specifies which files to search.
    """
    file_pattern = "**/" + file_pattern #create the file pattern so its arbitrary
    answers = []
    file_names = glob(file_pattern, recursive=True) #use glob to get all the file paths
    for file_name in file_names:
        with open(file_name) as f:
            contents = f.read() #initialize a contents
        if target_string in contents: #if the desired string is in the contents, add it to final list
            answers.append(file_name)
    return answers


# Problem 4
def largest_files(n):
    """Return a list of the n largest files in the current directory or its
    subdirectories (from largest to smallest).
    """
    filenames = glob("**/*.*", recursive=True)
    sizes = [[f,os.path.getsize(f)] for f in filenames]
    sizes.sort(key=lambda x:x[1], reverse=True) #key?
    list_files = [size[0] for size in sizes[:n]]
    smallest = list_files[-1]

    #full_names = []
    #sizes = []
    #for directory, subdirectories, files in os.walk('.'):
    #    for filename in files: #iterate through the filenames
    #        full_name = os.path.join(directory,filename) #new way
    #        #full_name = directory + "/" + filename (old way)
    #        full_names.append(full_name)
    #        size = int(subprocess.check_output(["ls","-s", full_name]).decode().rstrip().split()[0]) #get the file size
    #        sizes.append(size) #add the size
    #index_order = np.argsort(sizes)[::-1] #get descending order
    #full_names = np.array(full_names)
    #full_names = full_names[index_order] #reoder accordingly
    #sizes = full_names[index_order]

    #special_file = full_names[n]
    line_count = subprocess.check_output(["wc","-l", smallest]).decode() #format output
    line_count = line_count.rstrip().split() #format output
    line_count = line_count[0] #get the line count
    outfile = "smallest.txt"
    with open(outfile,mode='w') as f:
        f.write(line_count)

    return list_files #return the list of length n

# Problem 6
def prob6(n = 10):
   """this problem counts to or from n three different ways, and
      returns the resulting lists each integer

   Parameters:
       n (int): the integer to count to and down from
   Returns:
       integerCounter (list): list of integers from 0 to the number n
       twoCounter (list): list of integers created by counting down from n by two
       threeCounter (list): list of integers created by counting up to n by 3
   """
   #print what the program is doing
   integerCounter = list()
   twoCounter = list()
   threeCounter = list()
   counter = n
   for i in range(n+1):
       integerCounter.append(i)
       if (i % 2 == 0):
           twoCounter.append(counter - i)
       if (i % 3 == 0):
           threeCounter.append(i)
   #return relevant values
   return integerCounter, twoCounter, threeCounter
