# breadth_first_search.py
"""Volume 2: Breadth-First Search.
Sam Goldrup
MATH 321
28 October
"""

import itertools
import networkx as nx
from matplotlib import pyplot as plt
import numpy as np

# Problems 1-3
class Graph:
    """A graph object, stored as an adjacency dictionary. Each node in the
    graph is a key in the dictionary. The value of each key is a set of
    the corresponding node's neighbors.

    Attributes:
        d (dict): the adjacency dictionary of the graph.
    """
    def __init__(self, adjacency={}):
        """Store the adjacency dictionary as a class attribute"""
        self.d = dict(adjacency)

    def __str__(self):
        """String representation: a view of the adjacency dictionary."""
        return str(self.d)

    # Problem 1
    def add_node(self, n):
        """Add n to the graph (with no initial edges) if it is not already
        present.

        Parameters:
            n: the label for the new node.
        """
        if n not in self.d:
            self.d[n] = set()

    # Problem 1
    def add_edge(self, u, v):
        """Add an edge between node u and node v. Also add u and v to the graph
        if they are not already present.

        Parameters:
            u: a node label.
            v: a node label.
        """
        if u not in self.d: #if u isn't in it, make it
            self.add_node(u)
        if v not in self.d: #likewise for v
            self.add_node(v)
        self.d[u].add(v) #adds u to v's set
        self.d[v].add(u) #likewise

    # Problem 1
    def remove_node(self, n):
        """Remove n from the graph, including all edges adjacent to it.

        Parameters:
            n: the label for the node to remove.

        Raises:
            KeyError: if n is not in the graph.
        """
        if n in self.d:
            self.d.pop(n)
            for node in self.d:
                self.d[node].discard(n)
        else:
            raise KeyError("WHAT are you and idiot?! node not found")

    # Problem 1
    def remove_edge(self, u, v):
        """Remove the edge between nodes u and v.

        Parameters:
            u: a node label.
            v: a node label.

        Raises:
            KeyError: if u or v are not in the graph, or if there is no
                edge between u and v.
        """
        #the first 3 branches above might not be necessary
        if u not in self.d and v not in self.d:
            raise KeyError("neither" + u + "nor" + v + "are in the graph")
        elif u not in self.d:
            raise KeyError(u + "is not in the graph")
        elif v not in self.d:
            raise KeyError(v + "is not in the graph")
        elif u not in self.d[v] or v not in self.d[u]:
            raise KeyError("edge between" + u + "and" + v + "is not in the graph")

        self.d[u].discard(v)
        self.d[v].discard(u)

    # Problem 2
    def traverse(self, source):
        """Traverse the graph with a breadth-first search until all nodes
        have been visited. Return the list of nodes in the order that they
        were visited.

        Parameters:
            source: the node to start the search at.

        Returns:
            (list): the nodes in order of visitation.

        Raises:
            KeyError: if the source node is not in the graph.
        """
        V = [] #list of visited nodes
        Q = [] #"queue" of visited nodes in order they were discovered, might need to be a deque
        M = {} #set of nodes that have been visited or are marked TO be visited. a union of V and Q

        if source not in self.d:
            raise KeyError(source + "is not in the graph")

        Q.append(source)
        M = set.union(set(V), set(Q))

        while len(Q) != 0: #while nonempty
            just_visited = Q.pop(0) #pop first node
            V.append(just_visited) #add that node to V
            for item in self.d[just_visited]: #add neighbors to Q
                if item not in M: #not if its not in Q
                    Q.append(item) #add it Q
                M.add(item) #add it to M, no duplicates to take care of
        return V

    # Problem 3
    def shortest_path(self, source, target):
        """Begin a BFS at the source node and proceed until the target is
        found. Return a list containing the nodes in the shortest path from
        the source to the target, including endoints.

        Parameters:
            source: the node to start the search at.
            target: the node to search for.

        Returns:
            A list of nodes along the shortest path from source to target,
                including the endpoints.

        Raises:
            KeyError: if the source or target nodes are not in the graph.
        """
        V = [] #list of visited nodes
        Q = [] #"queue" of visited nodes in order they were discovered, might need to be a deque
        M = {} #set of nodes that have been visited or are marked TO be visited. a union of V and Q
        Preds = {}
        Path = []

        if source not in self.d or target not in self.d: #the nodes aren't in the graph
            raise KeyError("source or target not found")

        Q.append(source)
        M = set.union(set(V), set(Q))

        while len(Q) != 0: #while nonempty
            just_visited = Q.pop(0) #pop first node
            V.append(just_visited) #add that node to V
            for item in self.d[just_visited]: #add neighbors to Q
                if item not in M: #not if its not in Q
                    Q.append(item) #add it Q
                    Preds[item] = just_visited
                M.add(item) #add it to M, no duplicates to take care of
            


        if source == target:
            return list(source)

        pred = Preds[target] #might want to take care of an edge case with no graph, one node
        Path.append(pred)
        step = pred
        while pred != source:
            pred = Preds[step] #set pred equal to the value of the key "step" from Preds dict
            Path.append(pred) #add the pred to the path
            step = pred #reassign the step to the pred

        Path = Path[::-1]
        Path.append(target) #add on the end of the path
        return Path

# Problems 4-6
class MovieGraph:
    """Class for solving the Kevin Bacon problem with movie data from IMDb."""

    # Problem 4
    def __init__(self, filename="movie_data.txt"):
        """Initialize a set for movie titles, a set for actor names, and an
        empty NetworkX Graph, and store them as attributes. Read the speficied
        file line by line, adding the title to the set of movies and the cast
        members to the set of actors. Add an edge to the graph between the
        movie and each cast member.

        Each line of the file represents one movie: the title is listed first,
        then the cast members, with entries separated by a '/' character.
        For example, the line for 'The Dark Knight (2008)' starts with

        The Dark Knight (2008)/Christian Bale/Heath Ledger/Aaron Eckhart/...

        Any '/' characters in movie titles have been replaced with the
        vertical pipe character | (for example, Frost|Nixon (2008)).
        """
        self.filename = filename
        self.movie_titles = set()
        self.actor_names = set()
        self.graph = nx.Graph()

        with open(filename, 'r') as myfile:
            lines = myfile.readlines() #parse it out line by line

        for line in lines:
            line = line.strip() #remove the newline escape sequence #or strip().split('/')
            line = line.split('/') #parse it out baby
            self.movie_titles.add(line[0])
            for actor in line[1:]:
                self.actor_names.add(actor) #put the actors in the actor_names attribute/list
                self.graph.add_edge(line[0], actor) #put dem edges in bb!
    # Problem 5
    def path_to_actor(self, source, target):
        """Compute the shortest path from source to target and the degrees of
        separation between source and target.

        Returns:
            (list): a shortest path from source to target, including endpoints and movies.
            (int): the number of steps from source to target, excluding movies.
        """
        return nx.shortest_path(self.graph,source,target), nx.shortest_path_length(self.graph,source,target)//2

    # Problem 6
    def average_number(self, target):
        """Calculate the shortest path lengths of every actor to the target
        (not including movies). Plot the distribution of path lengths and
        return the average path length.

        Returns:
            (float): the average path length from actor to target.
        """
        distances= [] #intialize empty list of distances
        path_lengths = nx.shortest_path_length(self.graph,target)
        for actor in path_lengths.keys():
            if actor in self.actor_names:
                distances.append(path_lengths[actor] // 2) #for each actor get the target num and put it in distances

        #[path_legnth // 2 for actor, path_length in dict_all_path_lengths.itmes() if k in self.actors]

        plt.hist(np.array(distances), bins=[i-.5 for i in range(8)]) #initialize the bin bins
        plt.xlabel("distance from target actor")
        plt.ylabel("number of actors in H-wood")
        plt.title("actor-name numbers")
        plt.show()

        return np.mean(distances) #return the "kevin" number (if it's kevin bacon!!)
