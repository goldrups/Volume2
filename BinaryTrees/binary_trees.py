# binary_trees.py
"""Volume 2: Binary Trees.
Sam GOldrup
Math 321
7 October 2021
"""

# These imports are used in BST.draw().
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import time
import random
from matplotlib import pyplot as plt
import numpy as np


class SinglyLinkedListNode:
    """A node with a value and a reference to the next node."""
    def __init__(self, data):
        self.value, self.next = data, None

class SinglyLinkedList:
    """A singly linked list with a head and a tail."""
    def __init__(self):
        self.head, self.tail = None, None

    def append(self, data):
        """Add a node containing the data to the end of the list."""
        n = SinglyLinkedListNode(data)
        if self.head is None:
            self.head, self.tail = n, n
        else:
            self.tail.next = n
            self.tail = n

    def iterative_find(self, data):
        """Search iteratively for a node containing the data.
        If there is no such node in the list, including if the list is empty,
        raise a ValueError.

        Returns:
            (SinglyLinkedListNode): the node containing the data.
        """
        current = self.head
        while current is not None:
            if current.value == data:
                return current
            current = current.next
        raise ValueError(str(data) + " is not in the list")

    # Problem 1
    def recursive_find(self, data):
        """Search recursively for the node containing the data.
        If there is no such node in the list, including if the list is empty,
        raise a ValueError.

        Returns:
            (SinglyLinkedListNode): the node containing the data.
        """
        curr_node = self.head #starts the search at the head

        if curr_node is None:
            raise ValueError("list is empty") 

        def check_a_node(tha_node): 
            if tha_node is None: #this happens is the data is not in any of the keys of the nodes
                raise ValueError("data could not be found")
            elif tha_node.value == data:
                return tha_node
            else:
                return check_a_node(tha_node.next) #recursive call

        return check_a_node(curr_node)


class BSTNode:
    """A node class for binary search trees. Contains a value, a
    reference to the parent node, and references to two child nodes.
    """
    def __init__(self, data):
        """Construct a new node and set the value attribute. The other
        attributes will be set when the node is added to a tree.
        """
        self.value = data
        self.prev = None        # A reference to this node's parent node.
        self.left = None        # self.left.value < self.value
        self.right = None       # self.value < self.right.value


class BST:
    """Binary search tree data structure class.
    The root attribute references            if current is None:                     # Base case 1: dead end.
                raise ValueError(str(data) + " is not in the tree.")
            if data == current.value:               # Base case 2: data found!
                return current
            if data < current.value:                # Recursively search left.
                return _step(current.left)
            else:                                   # Recursively search right.
                return _step(current.right) the first node in the tree.
    """
    def __init__(self):
        """Initialize the root attribute."""
        self.root = None

    def find(self, data):
        """Return the node containing the data. If there is no such node
        in the tree, including if the tree is empty, raise a ValueError.
        """

        # Define a recursive function to traverse the tree.
        def _step(current):
            """Recursively step through the tree until the node containing
            the data is found. If there is no such node, raise a Value Error.
            """
            if current is None:                     # Base case 1: dead end.
                raise ValueError(str(data) + " is not in the tree.")
            if data == current.value:               # Base case 2: data found!
                return current
            if data < current.value:                # Recursively search left.
                return _step(current.left)
            else:                                   # Recursively search right.
                return _step(current.right)

        # Start the recursion on the root of the tree.
        return _step(self.root)

    # Problem 2
    def insert(self, data):
        """Insert a new node containing the specified data.

        Raises:
            ValueError: if the data is already in the tree.

        Example:
            >>> tree = BST()                    |
            >>> for i in [4, 3, 6, 5, 7, 8, 1]: |            (4)
            ...     tree.insert(i)              |            / \
            ...                                 |          (3) (6)
            >>> print(tree)                     |          /   / \
            [4]                                 |        (1) (5) (7)
            [3, 6]                              |                  \
            [1, 5, 7]                           |                  (8)
            [8]                                 |
        """
        if self.root is None: #tree is Empty
            new_node = BSTNode(data)
            self.root = new_node
        else:
            new_node = BSTNode(data)
            found_mom = False #bool for finding the parent node
            current = self.root #start the search at the root
            while found_mom == False:
                if data == current.value:
                    raise ValueError("data is already in tree")
                if data < current.value and current.left is None: #node belongs on left side AND there is a spot open
                    found_mom = True
                    current.left = new_node
                    new_node.prev = current
                elif data < current.value: #node belongs on left side BUT spot is taken
                    current = current.left
                if data > current.value and current.right is None: #node belongs on right side AND spot is open
                    found_mom = True
                    current.right = new_node
                    new_node.prev = current
                elif data > current.value: #node belongs on right side BUT spot is taken
                    current = current.right



    # Problem 3
    def remove(self, data):
        """Remove the node containing the specified data.

        Raises:
            ValueError: if there is no node containing the data, including if
                the tree is empty.

        Examples:
            >>> print(12)                       | >>> print(t3)
            [6]                                 | [5]
            [4, 8]                              | [3, 6]
            [1, 5, 7, 10]                       | [1, 4, 7]
            [3, 9]                              | [8]
            >>> for x in [7, 10, 1, 4, 3]:      | >>> for x in [8, 6, 3, 5]:
            ...     t1.remove(x)                | ...     t3.remove(x)
            ...                                 | ...
            >>> print(t1)                       | >>> print(t3)
            [6]                                 | [4]
            [5, 8]                              | [1, 7]
            [9]                                 |
                                                | >>> print(t4)
            >>> print(t2)                       | [5]
            [2]                                 | >>> t4.remove(1)
            [1, 3]                              | ValueError: <message>
            >>> for x in [2, 1, 3]:             | >>> t4.remove(5)
            ...     t2.remove(x)                | >>> print(t4)
            ...                                 | []
            >>> print(t2)                       | >>> t4.remove(5)
            []                                  | ValueError: <message>
        """
        target = self.find(data)
        if target.left is None and target.right is None: #leaf
            if self.root is target: #first check if it is the root
                self.root = None  
            elif target.prev.left is target: #if it is not the root, then check if it is a left child
                target.prev.left = None
            elif target.prev.right is target: #if it is not a left child, check if it is a right child (should always execute)
                target.prev.right = None  
        elif target.left is None or target.right is None: #it has kids
            if self.root is target: #first check if it is the root
                if self.root.left is None: #if it is the root, check if has a right child
                    self.root = self.root.right
                    self.root.prev = None 
                elif self.root.right is None: #if it doesn't have a right child, it must have a left child
                    self.root = self.root.left
                    self.root.prev = None
            elif target.left is None:
                #removal with one right child
                if target.value < target.prev.value: #the target is a left child
                    target.prev.left = target.right #sets target's parent's LEFT to target's child
                    target.right.prev = target.prev #sets child's PREV to target's parent
                if target.value > target.prev.value: #the target is a right child
                    target.prev.right = target.right #sets target's parent's RIGHT to target's child
                    target.right.prev = target.prev #sets child's PREV to to target's parent
            elif target.right is None:
                #remove node that has a left child
                if target.value < target.prev.value: #the target is a left child
                    target.prev.left = target.left #set target's parent's LEFT --> child
                    target.left.prev = target.prev #set child's PREV --> target's parent
                if target.value > target.prev.value: #the target is a right child
                    target.prev.right = target.left #set target's parent's RIGHT --> child
                    target.left.prev = target.prev #set child's PREV --> target's parent
        else: #target has two children
            curr_node = target.left #go one node to the left is outlined in the lab spec
            while curr_node.right is not None: #go to the right until it is a leaf
                curr_node = curr_node.right #step right
            insert_val = curr_node.value #copy the value of the leaf
            self.remove(curr_node.value) #just remove the leaf now
            target.value = insert_val #change the target's value to the leaf


    def __str__(self):
        """String representation: a hierarchical view of the BST.

        Example:  (3)
                  / \     '[3]          The nodes of the BST are printed
                (2) (5)    [2, 5]       by depth levels. Edges and empty
                /   / \    [1, 4, 6]'   nodes are not printed.
              (1) (4) (6)
        """
        if self.root is None:                       # Empty tree
            return "[]"
        out, current_level = [], [self.root]        # Nonempty tree
        while current_level:
            next_level, values = [], []
            for node in current_level:
                values.append(node.value)
                for child in [node.left, node.right]:
                    if child is not None:
                        next_level.append(child)
            out.append(values)
            current_level = next_level
        return "\n".join([str(x) for x in out])

    def draw(self):
        """Use NetworkX and Matplotlib to visualize the tree."""
        if self.root is None:
            return

        # Build the directed graph.
        G = nx.DiGraph()
        G.add_node(self.root.value)
        nodes = [self.root]
        while nodes:
            current = nodes.pop(0)
            for child in [current.left, current.right]:
                if child is not None:
                    G.add_edge(current.value, child.value)
                    nodes.append(child)

        # Plot the graph. This requires graphviz_layout (pygraphviz).
        nx.draw(G, pos=graphviz_layout(G, prog="dot"), arrows=True,
                with_labels=True, node_color="C1", font_size=8)
        plt.show()


class AVL(BST):
    """Adelson-Velsky Landis binary search tree data structure class.
    Rebalances after insertion when needed.
    """
    def insert(self, data):
        """Insert a node containing the data into the tree, then rebalance."""
        BST.insert(self, data)      # Insert the data like usual.
        n = self.find(data)
        while n:                    # Rebalance from the bottom up.
            n = self._rebalance(n).prev

    def remove(*args, **kwargs):
        """Disable remove() to keep the tree in balance."""
        raise NotImplementedError("remove() is disabled for this class")

    def _rebalance(self,n):
        """Rebalance the subtree starting at the specified node."""
        balance = AVL._balance_factor(n)
        if balance == -2:                                   # Left heavy
            if AVL._height(n.left.left) > AVL._height(n.left.right):
                n = self._rotate_left_left(n)                   # Left Left
            else:
                n = self._rotate_left_right(n)                  # Left Right
        elif balance == 2:                                  # Right heavy
            if AVL._height(n.right.right) > AVL._height(n.right.left):
                n = self._rotate_right_right(n)                 # Right Right
            else:
                n = self._rotate_right_left(n)                  # Right Left
        return n

    @staticmethod
    def _height(current):
        """Calculate the height of a given node by descending recursively until
        there are no further child nodes. Return the number of children in the
        longest chain down.
                                    node | height
        Example:  (c)                  a | 0
                  / \                  b | 1
                (b) (f)                c | 3
                /   / \                d | 1
              (a) (d) (g)              e | 0
                    \                  f | 2
                    (e)                g | 0
        """
        if current is None:     # Base case: the end of a branch.
            return -1           # Otherwise, descend down both branches.
        return 1 + max(AVL._height(current.right), AVL._height(current.left))

    @staticmethod
    def _balance_factor(n):
        return AVL._height(n.right) - AVL._height(n.left)

    def _rotate_left_left(self, n):
        temp = n.left
        n.left = temp.right
        if temp.right:
            temp.right.prev = n
        temp.right = n
        temp.prev = n.prev
        n.prev = temp
        if temp.prev:
            if temp.prev.value > temp.value:
                temp.prev.left = temp
            else:
                temp.prev.right = temp
        if n is self.root:
            self.root = temp
        return temp

    def _rotate_right_right(self, n):
        temp = n.right
        n.right = temp.left
        if temp.left:
            temp.left.prev = n
        temp.left = n
        temp.prev = n.prev
        n.prev = temp
        if temp.prev:
            if temp.prev.value > temp.value:
                temp.prev.left = temp
            else:
                temp.prev.right = temp
        if n is self.root:
            self.root = temp
        return temp

    def _rotate_left_right(self, n):
        temp1 = n.left
        temp2 = temp1.right
        temp1.right = temp2.left
        if temp2.left:
            temp2.left.prev = temp1
        temp2.prev = n
        temp2.left = temp1
        temp1.prev = temp2
        n.left = temp2
        return self._rotate_left_left(n)

    def _rotate_right_left(self, n):
        temp1 = n.right
        temp2 = temp1.left
        temp1.left = temp2.right
        if temp2.right:
            temp2.right.prev = temp1
        temp2.prev = n
        temp2.right = temp1
        temp1.prev = temp2
        n.right = temp2
        return self._rotate_right_right(n)


# Problem 4
def prob4():
    """Compare the build and search times of the SinglyLinkedList, BST, and
    AVL classes. For search times, use SinglyLinkedList.iterative_find(),
    BST.find(), and AVL.find() to search for 5 random elements in each
    structure. Plot the number of elements in the structure versus the build
    and search times. Use log scales where appropriate.
    """
    times_sll_load = [] #initialize empty lists of times
    times_bst_load = []
    times_avl_load = []

    times_sll_find = []
    times_bst_find = []
    times_avl_find = []

    with open("english.txt", "r") as file: #make a list of lines
        lines = file.readlines()

    n_vals = [2**k for k in range(3,11)] #different sizes 8,16,32,....
    for n in n_vals:
        random_sample = random.sample(lines, n) #creates a random sampling of n words from english.txt
        find_sample = random.sample(random_sample,5) #pick 5 of those 

        timed_sll = SinglyLinkedList() #make a SLL object
        time_sll_load_st = time.time() 
        for word in random_sample:
            timed_sll.append(word)
        time_sll_load_ed = time.time() 
        time_sll_load = time_sll_load_ed - time_sll_load_st #take difference of times

        timed_bst = BST() #make BST object
        time_bst_load_st = time.time()
        for word in random_sample:
            timed_bst.insert(word)
        time_bst_load = time.time() - time_bst_load_st
       
        timed_avl = AVL() #make AVL object
        time_avl_load_st = time.time()
        for word in random_sample:
            timed_avl.insert(word)
        time_avl_load = time.time() - time_avl_load_st
        
        #now we time finding
        time_sll_find_st = time.time() 
        for word in find_sample:
            timed_sll.iterative_find(word)
        time_sll_find = time.time() - time_sll_find_st

        time_bst_find_st = time.time()
        for word in find_sample:
            timed_bst.find(word)
        time_bst_find = time.time() - time_bst_find_st

        time_avl_find_st = time.time()
        for word in find_sample:
            timed_avl.find(word)
        time_avl_find = time.time() - time_avl_find_st
        
        times_sll_load.append(time_sll_load) #append the times to the list of times
        times_bst_load.append(time_bst_load)
        times_avl_load.append(time_avl_load)

        times_sll_find.append(time_sll_find)
        times_bst_find.append(time_bst_find)
        times_avl_find.append(time_avl_find)



    plt.subplot(121).loglog(n_vals, times_sll_load, 'b.-', label = "SLL")
    plt.subplot(121).loglog(n_vals, times_bst_load, 'r.-', label = "BST")
    plt.subplot(121).loglog(n_vals, times_avl_load, 'g.-', label = "AVL")
    plt.subplot(121).set_xlabel("size")
    plt.subplot(121).set_ylabel("load time")
    plt.title("loading")

    plt.subplot(122).loglog(n_vals, times_sll_find, 'b.-', label = "SLL")
    plt.subplot(122).loglog(n_vals, times_bst_find,  'r.-', label = "BST")
    plt.subplot(122).loglog(n_vals, times_avl_find, 'g.-', label = "AVL")
    plt.subplot(122).set_xlabel("size")
    plt.subplot(122).set_ylabel("find time")
    plt.title("finding")

    plt.suptitle("load and find times (log scale")
    plt.tight_layout() #if i'm gonna go hard on the lab imma make it look sexy --kendrick lamar
    plt.show()



            

