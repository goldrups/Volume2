# linked_lists.py
"""Volume 2: Linked Lists.
Samuel Goldrup
Math 321
7 October 2021
"""


# Problem 1
class Node:
    """A basic node class for storing data."""
    def __init__(self, data):
        """Store the data in the value attribute.

        Raises:
            TypeError: if data is not of type int, float, or str.
        """
        if type(data) != int and type(data) != float and type(data) != str:
            raise TypeError("invalid type for node. must be int, float or str")
        self.value = data


class LinkedListNode(Node):
    """A node class for doubly linked lists. Inherits from the Node class.
    Contains references to the next and previous nodes in the linked list.
    """
    def __init__(self, data):
        """Store the data in the value attribute and initialize
        attributes for the next and previous nodes in the list.
        """
        Node.__init__(self, data)       # Use inheritance to set self.value.
        self.next = None                # Reference to the next node.
        self.prev = None                # Reference to the previous node.


# Problems 2-5
class LinkedList:
    """Doubly linked list data structure class.

    Attributes:
        head (LinkedListNode): the first node in the list.
        tail (LinkedListNode): the last node in the list.
    """
    def __init__(self):
        """Initialize the head and tail attributes by setting
        them to None, since the list is empty initially.
        """
        self.head = None
        self.tail = None
        self.length = 0

    def append(self, data):
        """Append a new node containing the data to the end of the list."""
        # Create a new node to store the input data.
        new_node = LinkedListNode(data)
        if self.head is None:
            # If the list is empty, assign the head and tail attributes to
            # new_node, since it becomes the first and last node in the list.
            self.head = new_node
            self.tail = new_node
        else: #this is what executes
            # If the list is not empty, place new_node after the tail.
            self.tail.next = new_node               # tail --> new_node
            new_node.prev = self.tail               # tail <-- new_node
            # Now the last node in the list is new_node, so reassign the tail.
            self.tail = new_node
        self.length += 1


    # Problem 2
    def find(self, data): #not raising value error
        """Return the first node in the list containing the data.

        Raises:
            ValueError: if the list does not contain the data.

        Examples:
            >>> l = LinkedList()
            >>> for x in ['a', 'b', 'c', 'd', 'e']:
            ...     l.append(x)
            ...
            >>> node = l.find('b')
            >>> node.value
            'b'
            >>> l.find('f')
            ValueError: <message>
        """

        current_node = self.head
        if current_node is None:
            raise ValueError("list is empty")
        while True:
            if current_node.value == data:
                return current_node #exits loop if desired node is found
            elif current_node.next is None:
                raise ValueError("not in the list")
            else: #keeps going otherwise
                current_node = current_node.next


    # Problem 2
    def get(self, i):
        """Return the i-th node in the list.

        Raises:
            IndexError: if i is negative or greater than or equal to the
                current number of nodes.

        Examples:
            >>> l = LinkedList()
            >>> for x in ['a', 'b', 'c', 'd', 'e']:none
            ...     l.append(x)
            ...
            >>> node = l.get(3)
            >>> node.value
            'd'
            >>> l.get(5)
            IndexError: <message>
        """
        if i < 0:
            raise IndexError("negative index!!") #valueerror if negative input
        if i >= self.length:
            raise IndexError("out of index access") #value error if input too big
        current_node = self.head
        for j in range(i):
            current_node = current_node.next #iterates through list to the desired node
        return current_node


    # Problem 3
    def __len__(self):
        """Return the number of nodes in the list.

        Examples:
            >>> l = LinkedList()
            >>> for i in (1, 3, 5):
            ...     l.append(i)
            ...
            >>> len(l)
            3
            >>> l.append(7)
            >>> len(l)
            4
        """
        return self.length

    # Problem 3
    def __str__(self):
        """String representation: the same as a standard Python list.

        Examples:
            >>> l1 = LinkedList()       |   >>> l2 = LinkedList()
            >>> for i in [1,3,5]:       |   >>> for i in ['a','b',"c"]:
            ...     l1.append(i)        |   ...     l2.append(i)
            ...                         |   ...
            >>> print(l1)               |   >>> print(l2)
            [1, 3, 5]                   |   ['a', 'b', 'c']
        """
        current_node = self.head
        special_list = "["
        while current_node is not None:
            special_list += repr(current_node.value)
            if current_node.next is not None:
                special_list += ", " #add a comma after value if it is not last element
            current_node = current_node.next
        special_list += "]"
        return special_list

    # Problem 4
    def remove(self, data):
        """Remove the first node in the list containing the data.

        Raises:
            ValueError: if the list is empty or does not contain the data.

        Examples:
            >>> print(l1)               |   >>> print(l2)
            ['a', 'e', 'i', 'o', 'u']   |   [2, 4, 6, 8]
            >>> l1.remove('i')          |   >>> l2.remove(10)
            >>> l1.remove('a')          |   ValueError: <message>
            >>> l1.remove('u')          |   >>> l3 = LinkedList()
            >>> print(l1)               |   >>> l3.remove(10)
            ['e', 'o']                  |   ValueError: <message>
        """
        #now i need to consider the edge cases where we remove the first and last
        if self.head is None:
            raise ValueError("the list is empty") #ValueError if try to remove from empty list
        target = self.find(data)
        if target.prev is None and target.next is None: #if it is the only element in the list
            self.head = None
            self.tail = None
        elif target.prev is None: #if it is the head
            self.head = self.head.next
            self.head.prev.next = None
            self.head.prev = None
        elif target.next is None: #if it is the tail
            self.tail = self.tail.prev
            self.tail.next.prev = None
            self.tail.next = None
        else: #if it is somewhere in the middle
            target.prev.next = target.next
            target.next.prev = target.prev
            target.next = None
            target.prev = None
        self.length -= 1

    # Problem 5
    def insert(self, index, data):
        """Insert a node containing data into the list immediately before the
        node at the index-th location.

        Raises:
            IndexError: if index is negative or strictly greater than the
                current number of nodes.

        Examples:
            >>> print(l1)               |   >>> len(l2)
            ['b']                       |   5
            >>> l1.insert(0, 'a')       |   >>> l2.insert(6, 'z')
            >>> print(l1)               |   IndexError: <message>
            ['a', 'b']                  |
            >>> l1.insert(2, 'd')       |   >>> l3 = LinkedList()
            >>> print(l1)               |   >>> l3.insert(1, 'a')
            ['a', 'b', 'd']             |   IndexError: <message>
            >>> l1.insert(2, 'c')       |
            >>> print(l1)               |
            ['a', 'b', 'c', 'd']        |        if target.prev is None and target.next is None:

        """
        if index < 0:
            raise IndexError("index out of range") #valueerror if input too small
        elif index > self.length:
            raise IndexError("index out of range") #valueerror if input too big
        new_node = LinkedListNode(data)
        if self.head is None: #if list is empty
            self.append(data)
        elif index == self.length: #if appending to end of list
            self.append(data)
        elif index == 0 and self.length > 0: #if appending to beginning of nonempty list
            target = self.get(index)
            new_node = LinkedListNode(data)
            self.head.prev = new_node
            new_node.next = self.head
            self.head = new_node
            self.length += 1
        else: #if appending to somewhere in the middle of the list
            new_node = LinkedListNode(data)
            target = self.get(index)
            target.prev.next = new_node
            target.prev = new_node
            new_node.next = target
            new_node.prev = target.prev
            self.length += 1

# Problem 6: Deque class.
class Deque(LinkedList):
    """
    Makes a Deque using methods from LinkedList class
    overrides insert() and remove() so that popping and pushing only
    happens at the ends

    pop() pops from right side, popleft() pops from left end
    appendleft() appends to left side, append() is already written
    before these lines, so it is not written here
    """

    def __init__(self):
        """
        constructor inherits from LinkedList class
        """
        LinkedList.__init__(self)

    def pop(self):
        """
        raises value error if list is empty
        pops the value on the right end
        """
        if self.length == 0: #raise an error if the list is empty
            raise ValueError("list is empty")
        pop_val = self.tail.value
        LinkedList.remove(self, pop_val)
        return pop_val


    def popleft(self):
        """
        same as pop(), but for the left end
        """
        if self.length == 0: #raise and error if the list is empty
            raise ValueError("list is empty")
        pop_left_val = self.head.value
        LinkedList.remove(self, pop_left_val)
        return pop_left_val

    def appendleft(self, data):
        """
        appends the data to the left end of the deque
        """
        if self.head is None: #makes the list nonempty to take care of infinite recursion
            self.head = LinkedListNode(data)
            self.tail = LinkedListNode(data)
            self.length += 1
        else:
            LinkedList.insert(self, 0, data)


    #overrides the remove function
    def remove(*args, **kwargs):
        raise NotImplementedError("Use pop() or popleft() for removal")
        #now the user cannot remove a node from the middle of a list

    #overrides the insert function
    def insert(*args, **kwargs):
        raise NotImplementedError("Use appendleft() for adding")
        #now the user cannot inser a node to the middle of a list




# Problem 7
def prob7(infile, outfile):
    """Reverse the contents of a file by line and write the results to
    another file.

    Parameters:
        infile (str): the file to read from.
        outfile (str): the file to write to.
    """
    storage_deck = Deque() #implement the deque

    #open the file
    with open(infile,"r") as file:
        lines = file.readlines()

    #add newline to line if not currently there, store each line in the deque
    for line in lines:
        if line[-1] != '\n':
            line += '\n'
        storage_deck.appendleft(line)

    #write the lines to the desired file, now in the reversed order
    with open(outfile, "w") as file:
        for i in range(storage_deck.length):
            line = storage_deck.popleft()
            file.write(line)
