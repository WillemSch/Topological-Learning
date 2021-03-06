import numpy as np
from util import add_tuples
from util import multiply_tuple


def simple_neighbourhood_function(depth):
    """A simple example neighbourhood weight function for use with the SOM class.

    :param depth: The depth of the neighbourhood to get the weight for (Integer).
    :return: A weight for the neighbourhood at a given depth (1 if depth is 1, 0.2 if depth is 2, 0 otherwise).
    """
    weights = [1., .2]
    if depth > 2:
        return 0.
    else:
        return weights[depth]


class SOM:
    """A class to apply the Self-Organising-Map algorithm over a given graph.

    :param graph: The graph to apply the Self-Organising-Map algorithm over, (type Graph from this package)
    :param neighbourhood_function: Optional, default = None - The neighbourhood weight function, takes in depth
        (integer) and returns the weight for that given depth (float). If None simple_neighbourhood_function is
        used.
    :param neighbourhood_depth: Optional, default = 2 - The maximum neighbourhood depth to consider.
    :param learning_rate: Optional, default = 0.1 - The rate at which the graph updates over each iteration.
    :param iterations: Optional, default = 10 - The amount of iteration the SOM algorithm will take.
    """

    def __init__(self, graph, neighbourhood_function=None, neighbourhood_depth=2, learning_rate=0.1, iterations=10):
        """Initializes the SOM class.

        :param graph: The graph to apply the Self-Organising-Map algorithm over, (type Graph from this package)
        :param neighbourhood_function: Optional, default = None - The neighbourhood weight function, takes in depth
            (integer) and returns the weight for that given depth (float). If None simple_neighbourhood_function is
            used.
        :param neighbourhood_depth: Optional, default = 2 - The maximum neighbourhood depth to consider.
        :param learning_rate: Optional, default = 0.1 - The rate at which the graph updates over each iteration.
        :param iterations: Optional, default = 10 - The amount of iteration the SOM algorithm will take.
        """
        self.graph = graph
        self.neighbourhood_function = neighbourhood_function
        self.neighbourhood_depth = neighbourhood_depth
        self.learning_rate = learning_rate
        self.iterations = iterations

    def fit(self, data):
        """Fits the SOM class to the given dataset.

        :param data: List of coordinate tuples.
        :return: This class instance.
        """
        if self.neighbourhood_function is None:
            self.neighbourhood_function = simple_neighbourhood_function
        return self

    def transform(self, data):
        """Applies the self-organising-map algorithm over a given dataset and graph. The initially given graph will be
        altered after running transform().

        :param data: List of coordinate tuples.
        :return: The altered graph.
        """
        for _ in range(self.iterations):
            for x in data:
                # Get closest node to the data-point
                closest = None
                dist = np.inf
                for n in self.graph.nodes.flatten():
                    neg = multiply_tuple(-1, x)
                    norm = np.linalg.norm(add_tuples(n.coordinates, neg))
                    if norm < dist:
                        dist = norm
                        closest = n

                for depth in range(self.neighbourhood_depth):
                    for n in closest.get_nth_neighbours(depth):
                        neg = multiply_tuple(-1, n.coordinates)
                        n.coordinates = add_tuples(n.coordinates,
                                                   multiply_tuple(self.learning_rate * self.neighbourhood_function(depth),
                                                                  (add_tuples(x, neg)))
                                                   )
        return self.graph
