import graph
import numpy as np
from util import multiply_tuple
from util import add_tuples
from util import average_tuples


class GNG:
    """A class to apply the Growing-Neural-Gas algorithm over a given dataset, generating a Graph.

    :param data: A list of coordinate tuples of data-points
    """

    def __init__(self, data):
        """Initializes the GNG class and generates a Graph to start the algorithm with.

        :param data: A list of coordinate tuples of data-points
        """

        # Keep the same amount of dimensions in the graph as the data-set has
        self.graph = graph.create_grid(2, dimensions=len(data.shape))
        self.data = data

    def train(self, iterations, learning_rate=0.1, age_threshold=100, node_creation_interval=100,
              new_node_error_discount=.5, step_error_discount=.1):
        """Applies the self-organising-map algorithm over a given dataset and graph. Returns a graph after running the
        algorithm. NOTE: Running train() multiple times will continue updating the same graph, to restart fresh
        create a new GNG instance.

        :param iterations: The amount of iterations to run the GNG algorithm
        :param learning_rate: Optional, default = 0.1 - The rate at which coordinates of nodes are updated towards the
            learned direction
        :param age_threshold: Optional, default = 100 - The maximum age an edge can reach before being removed.
        :param node_creation_interval: Optional, default = 100 - Defines at what amount of steps a new node should be
            added.
        :param new_node_error_discount: Optional default = 0.5 - The discount value for the error when a new node is
            created.
        :param step_error_discount: Optional default = 0.1 - The discount value for the error which is subtracted from
            the errors each step.
        :return: The Graph of the GNG class after running the GNG algorithm.
        """
        flattened = self.graph.nodes.flatten()
        error = {n: 0 for n in flattened}
        edges = {(flattened[edge[0]], flattened[edge[1]]): 0 for edge in self.graph.get_edges()} # Keeps track of the age of edges
        for count in range(iterations):
            for i_x, x in enumerate(self.data):
                i, j = self.__find_n_closest(x, n=2)
                flattened = self.graph.nodes.flatten()
                ni = flattened[i[0]]

                # Update error e_i
                error[ni] += i[1]

                # Update w_i
                ni.coordinates = add_tuples(ni.coordinates, multiply_tuple(
                    learning_rate,
                    add_tuples(x, multiply_tuple(-1, ni.coordinates))
                ))

                # Update w_j
                nj = flattened[j[0]]
                nj.coordinates = add_tuples(nj.coordinates, multiply_tuple(
                    learning_rate,
                    add_tuples(x, multiply_tuple(-1, nj.coordinates))
                ))

                # increment age of connected edges
                for edge in edges:
                    if ni in edge:
                        edges[edge] += 1

                # Add edge between i and j, if it doesn't exist
                new_edge = (min(i[0], j[0]), max(i[0], j[0]))
                if new_edge not in edges:
                    ni.connect(nj)
                    edges[new_edge] = 0

                # Remove old edges, and unconnected nodes
                removed_edges = []
                for edge in edges:
                    if edges[edge] > age_threshold:
                        edge[0].disconnect(edge[1])
                        removed_edges.append(edge)
                        # Remove any nodes that have no neighbours (no edges)
                        if len(edge[0].neighbours) == 0:
                            self.graph.remove_node(edge[0])
                        if len(edge[1].neighbours) == 0:
                            self.graph.remove_node(edge[1])
                for edge in removed_edges:
                    edges.pop(edge)

                # Every m steps insert node between nodes with highest error, and neighbour with highest error
                if ((count * len(self.data)) + i_x) % node_creation_interval == 0:
                    highest_error = max(error, key=error.get)
                    highest_neighbour = None
                    for n in highest_error.neighbours:
                        if not highest_neighbour or error[highest_neighbour] < error[n]:
                            highest_neighbour = n

                    # Create a new node on the average of the coordinates of the highest error and highest error
                    # neighbour
                    new = self.graph.add_node(average_tuples(highest_neighbour.coordinates, highest_error.coordinates))
                    highest_neighbour.disconnect(highest_error)
                    new.connect(highest_neighbour)
                    new.connect(highest_error)
                    error[highest_neighbour] -= new_node_error_discount
                    error[highest_error] -= new_node_error_discount
                    error[new] = 0

                # Decrease all errors
                for n in error:
                    error[n] -= step_error_discount
        return self.graph

    def __find_n_closest(self, x, n):
        """Find the closest N nodes to a given data-point in the graph of the GNG class

        :param x: A tuple of coordinates to find the closest nodes to.
        :param n: The amount of Nodes to look for.
        :return: A list of tuples of the closest N nodes. Tuples of form: (Node index, Distance to x)
        """
        dist = [(-(i + 1), np.inf) for i in range(n)]
        for index, n in enumerate(self.graph.nodes.flatten()):
            neg = multiply_tuple(-1, x)
            norm = np.linalg.norm(add_tuples(n.coordinates, neg))
            furthest = max(dist, key=lambda d: d[1])
            if norm < furthest[1]:
                dist.remove(furthest)
                dist.append((index, norm))
        return dist

