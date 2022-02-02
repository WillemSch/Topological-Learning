import numpy as np
from util import add_tuples
from util import multiply_tuple

def simple_neighbourhood_function(depth):
    weights = [1., .2]
    if depth > 2:
        return 0.
    else:
        return weights[depth]


class SOM:
    def __init__(self, graph):
        self.graph = graph

    def fit(self, data, neighbourhood_function=None, neighbourhood_depth=2, learning_rate=0.1, iterations=10):
        if neighbourhood_function is None:
            neighbourhood_function = simple_neighbourhood_function

        for _ in range(iterations):
            for x in data:
                closest = None
                dist = np.inf
                for n in self.graph.nodes.flatten():
                    neg = multiply_tuple(-1, x)
                    norm = np.linalg.norm(add_tuples(n.coordinates, neg))
                    if norm < dist:
                        dist = norm
                        closest = n

                for depth in range(neighbourhood_depth):
                    for n in closest.get_nth_neighbours(depth):
                        neg = multiply_tuple(-1, n.coordinates)
                        n.coordinates = add_tuples(n.coordinates,
                                                   multiply_tuple(learning_rate * neighbourhood_function(depth),
                                                                  (add_tuples(x, neg)))
                                                   )
