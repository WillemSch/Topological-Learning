import numpy as np
from util import add_tuples
from util import multiply_tuple


class Node:
    """A class for Nodes in a Graph, keeps track of neighbours, and coordinates.

    :param coordinates: The coordinates of this Node, should be a tuple or list of any dimension.
    """

    def __init__(self, coordinates):
        """Initializes the Node class. Sets its coordinates to the given coordinates, and sets its neighbours to an
        empty list.

        :param coordinates: The coordinates of this Node, should be a tuple or list of any dimension.
        """
        self.key_coords = tuple(coordinates)
        self.neighbours = []
        self.coordinates = tuple(coordinates)

    def __key(self):
        """Create __key function to use initial_coordinate as key for class Node.

        :return: The initial coordinates of this node.
        """
        return self.key_coords

    def __hash__(self):
        """Overrides the __hash__() function and creates a hash based on __key.

        :return: A hash for this Node.
        """
        return hash(self.__key())

    def __eq__(self, other):
        """Overrides the __eq__() function (Equals).

        :param other: The object to compate to.
        :return: True if other is of class Node with the same __key, False if other is of class Node with different
            __key, otherwise throws NotImplemented.
        """
        if isinstance(other, Node):
            return self.__key() == other.__key()
        return NotImplemented

    def connect(self, other):
        """Connect this node to the other Node. Will add other to this node's neighbourhood, and this node to the
        other's neighbourhood.

        :param other: The Node to connect this node to.
        :return: None
        """
        if other not in self.neighbours:
            self.neighbours.append(other)
        if self not in other.neighbours:
            other.neighbours.append(self)

    def disconnect(self, other):
        """Opposite of Connect(), removes this node and other from each other's neighbourhood.

        :param other: The Node to disconnect from this node.
        :return: None
        """
        if other in self.neighbours:
            self.neighbours.remove(other)
        if self in other.neighbours:
            other.neighbours.remove(self)

    def get_nth_neighbours(self, depth):
        """Get all nodes that are neighbour of this node at a given depth

        :param depth: The depth of the neighbouring nodes to get.
        :return: A list of Nodes that are neighbour of this node at the given depth. Returns [self] if depth = 0.
        """
        neighbours = [self]
        tmp = [self]
        old = []
        for i in range(depth):
            old.extend(set(neighbours))  # Keep track of previously visited nodes, so we don't go in circles
            neighbours = []
            for n in tmp:
                neighbours.extend(n.neighbours)
            neighbours = set(neighbours)  # Trick to remove duplicates
            neighbours = [x for x in neighbours if x not in old]
            tmp = neighbours
        return neighbours


class Graph:
    """A class for a Graph keeps track of a collection of nodes.

    :param shape: The shape of the collection of Nodes (Tuple).
    :param dimensions: Optional, default None - The amount of dimensions a node uses for its coordinates. If none is
        given dimensions = len(shape).
    :param coord_scale: Optional, default 1 - Nodes are initialized with unique coordinates based on their index,
        coord_scale is a scalar for these initial coordinates.
    """

    def __init__(self, shape, dimensions=None, coord_scale=1):
        """Initializes the Graph class.

        :param shape: The shape of the collection of Nodes (Tuple).
        :param dimensions: Optional, default None - The amount of dimensions a node uses for its coordinates. If none is
            given dimensions = len(shape).
        :param coord_scale: Optional, default 1 - Nodes are initialized with unique coordinates based on their index,
            coord_scale is a scalar for these initial coordinates.
        """
        if dimensions is None:
            dimensions = len(shape)

        self.nodes = np.ndarray(shape=shape, dtype=np.object)
        for i, _ in np.ndenumerate(self.nodes):
            coords = add_tuples(tuple(np.zeros(dimensions)), i)
            self.nodes[i] = Node(coordinates=multiply_tuple(coord_scale, coords))

    def add_node(self, coordinates):
        """Adds a node to the graph with given coordinates.

        :param coordinates: A tuple with the coordinates of the Node.
        :return: The new node, of class Node.
        """
        assert(len(self.nodes.shape) == 1)
        new_node = Node(coordinates)
        self.nodes = np.append(self.nodes, [new_node])
        return new_node

    def remove_node(self, node):
        """Removes a given node from the graph.

        :param node: The Node to be removed.
        :return: None
        """
        for n in node.neighbours:
            node.disconnect(n)
        index = np.where(self.nodes == node)
        np.delete(self.nodes, index)

    def get_edges(self):
        """Get a list of all edges in the Graph

        :return: A list of Tuples with the indices of connected Nodes.
        """
        flattened = list(self.nodes.flatten())
        edges = set()
        for i, node in enumerate(flattened):
            for neighbour in node.neighbours:
                neighbour_index = flattened.index(neighbour)
                edges.add((min(i, neighbour_index), max(i, neighbour_index)))  # Sort indices so that we don't generate duplicate edges.
        return edges


def create_grid(shape, coord_scale=1, dimensions=None):
    """Create a square-grid graph of a given shape. Where all nodes that are next to each other are connected.

    :param shape: The shape of the grid graph to be created.
    :param coord_scale: Optional, default 1 - Nodes are initialized with unique coordinates based on their index,
        coord_scale is a scalar for these initial coordinates.
    :param dimensions: Optional, default None - The amount of dimensions a node uses for its coordinates. If none is
        given dimensions = len(shape).
    :return: A Graph containing nodes in the grid shape, with neighbours connected.
    """
    g = Graph(shape, coord_scale=coord_scale, dimensions=dimensions)
    for i, _ in np.ndenumerate(g.nodes):
        connect_to_neighbours(i, g)
    return g


def connect_to_neighbours(index, graph):
    """Used to create a square-grid graph. Connects a node to all its direct neighbours in all dimensions based on the
    index of nodes.

    :param index: The index of the node that should be connected to its neighbouring nodes.
    :param graph: The graph containing the nodes.
    :return: None
    """
    node = graph.nodes[index]
    for i, d in enumerate(index):
        if d > 0:
            neighbour_index = list(index).copy()
            neighbour_index[i] -= 1
            neighbour = graph.nodes[tuple(neighbour_index)]
            node.connect(neighbour)
        if d < graph.nodes.shape[i] - 1:
            neighbour_index = list(index).copy()
            neighbour_index[i] += 1
            neighbour = graph.nodes[tuple(neighbour_index)]
            node.connect(neighbour)

