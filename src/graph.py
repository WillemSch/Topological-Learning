import numpy as np
# import networkx as nx
from util import add_tuples
from util import multiply_tuple
from util import average_tuples


class Node:
    def __init__(self, coordinates):
        self.key_coords = tuple(coordinates)
        self.neighbours = []
        self.coordinates = tuple(coordinates)

    def __key(self):
        return self.key_coords

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.__key() == other.__key()
        return NotImplemented

    def connect(self, other):
        if other not in self.neighbours:
            self.neighbours.append(other)
        if self not in other.neighbours:
            other.neighbours.append(self)

    def disconnect(self, other):
        if other in self.neighbours:
            self.neighbours.remove(other)
        if self in other.neighbours:
            other.neighbours.remove(self)

    def get_nth_neighbours(self, depth):
        neighbours = [self]
        tmp = [self]
        old = []
        for i in range(depth):
            old.extend(set(neighbours))
            neighbours = []
            for n in tmp:
                neighbours.extend(n.neighbours)
            neighbours = set(neighbours)
            neighbours = [x for x in neighbours if x not in old]
            tmp = neighbours
        return neighbours


class Graph:
    def __init__(self, shape, dimensions=None, coord_scale=1):
        if dimensions is None:
            dimensions = len(shape)

        self.nodes = np.ndarray(shape=shape, dtype=np.object)
        for i, _ in np.ndenumerate(self.nodes):
            coords = add_tuples(tuple(np.zeros(dimensions)), i)
            self.nodes[i] = Node(coordinates=multiply_tuple(coord_scale, coords))

    def add_node(self, coordinates):
        assert(len(self.nodes.shape) == 1)
        new_node = Node(coordinates)
        self.nodes = np.append(self.nodes, [new_node])
        return new_node

    def remove_node(self, node):
        for n in node.neighbours:
            node.disconnect(n)
        index = np.where(self.nodes == node)
        np.delete(self.nodes, index)

    def get_edges(self):
        flattened = list(self.nodes.flatten())
        edges = set()
        for i, node in enumerate(flattened):
            for neighbour in node.neighbours:
                neighbour_index = flattened.index(neighbour)
                edges.add((min(i, neighbour_index), max(i, neighbour_index)))
        return edges

    # def to_nx_graph(self):
    #     export = nx.Graph()
    #     added_nodes = []
    #     for node in self.nodes.flatten():
    #         export.add_node(node, pos=node.coordinates)
    #         added_nodes.append(node)
    #     for edge in self.get_edges():
    #         export.add_edge(added_nodes[edge[0]], added_nodes[edge[1]])
    #     return export


def create_grid(shape, coord_scale=1, dimensions=None):
    g = Graph(shape, coord_scale=coord_scale, dimensions=dimensions)
    for i, _ in np.ndenumerate(g.nodes):
        connect_to_neighbours(i, g)
    return g


def connect_to_neighbours(index, graph):
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

