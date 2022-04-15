import numpy
import graph as Graph
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np


def first_dimension(x):
    """Get the value of the first dimension of x. Can be used as projection function for reeb graphs.

    :param x: A datapoint
    :return: the value of the first dimension of x - (x[0])
    """
    return x[0]


def second_dimension(x):
    """Get the value of the second dimension of x. Can be used as projection function for reeb graphs.

    :param x: A datapoint
    :return: the value of the second dimension of x - (x[1])
    """
    return x[1]


class Reeb:
    """A class to generate a Reeb graph for a given dataset and projection function
    """

    # function has to map data to a real number
    def __init__(self, function=second_dimension):
        """Initializes the Reeb class, with a projection function and creates a Graph instance to be used.

        :param function: Optional, default second_dimension() - A function that maps a datapoint to a number.
        """
        self.function = function
        self.g = Graph.Graph(0, dimensions=2)

    def map(self, data, intervals, overlap=.2, max_k=5):
        """Generate a Reeb-graph for a given dataset using the pre-defined projection function.

        :param data: A list of datapoints before they are passed through the projection function.
        :param intervals: The amount of intervals the reeb-algorithm will divide the range of points in.
        :param overlap: Optional, default 0.2 - The overlap of the intervals, in percentages (1 = 100%, .5 = 50%, etc.)
        :param max_k: The maximum amount of clusters per interval that are tested. Limits nodes per interval to the range 1 to max_k.
        :return: A Reeb-graph fitted to the given dataset with given parameters.
        """
        processed_data = [[self.function(x)] for x in data]
        data_range = (np.min(processed_data), np.max(processed_data))
        range_distance = np.abs(data_range[1] - data_range[0])
        interval_size = (range_distance * (1 + overlap)) / intervals

        interval_end = interval_size * overlap + data_range[0]
        prev_nodes = []
        for i in range(intervals):
            interval_start = interval_end - interval_size * overlap
            interval_end = interval_start + interval_size
            interval = (interval_start, interval_end)
            preimage_data = self.__get_preimage_for_interval(data, processed_data, interval)

            # Since silhouette score can only be computed for k >= 2, we say K = 1 is best if no other k gets a
            # silhouette score above .6
            best_cluster_score = (np.zeros(len(preimage_data)), .6, 1)
            for k in range(2, max_k + 1):
                if k > len(preimage_data) - 1:
                    break
                km = KMeans(n_clusters=k)
                prediction = km.fit_predict(preimage_data)
                score = silhouette_score(preimage_data, prediction)
                if score > best_cluster_score[1]:
                    best_cluster_score = (prediction, score, k)

            new_nodes = []
            for k in range(best_cluster_score[2]):
                new_node = self.g.add_node((k, (interval[0] + interval[1]) / 2))
                node_data = np.array([preimage_data[i] for i, x in enumerate(best_cluster_score[0]) if x == k])
                for old in prev_nodes:
                    if self.__has_overlapping_data(node_data, old[1]):
                        new_node.connect(old[0])
                new_nodes.append((new_node, node_data))
            prev_nodes = new_nodes

        self.rearrange_nodes()
        return self.g

    def __get_preimage_for_interval(self, data, processed_data, interval):
        """Returns the preimage of a processed data-set over a given interval.

        :param data: List of data-points; The original dataset.
        :param processed_data: List of data-points from the original dataset processed by some function.
        :param interval: Tuple indicating the interval to get the preimage over. Inclusive for both low and high if interval = (low, high).
        :return: List of data-points from the original dataset that match with processed data-points in a given interval.
        """
        return np.array([data[i] for i, x in enumerate(processed_data) if interval[0] <= x[0] <= interval[1]])

    def __has_overlapping_data(self, a_data, b_data, threshold=1):
        """Check if there is a non-empty intersection between 2 data sets.

        :param a_data: NdArray of points.
        :param b_data: NdArray of points.
        :param threshold: Optional, default 1 - The minimum amount of overlapping points.
        :return: True if the intersection between a anb b is bigger or equals to the threshold, False otherwise.
        """
        return len(numpy.intersect1d(a_data.flatten(), b_data.flatten())) >= threshold

    def rearrange_nodes(self):
        """Rearrange the nodes in the graph of this instance to make it a bit easier to look at when visualizing.

        :return: None.
        """
        graphs = []
        to_be_processed = self.g.nodes.flatten()

        while len(to_be_processed) > 0:
            node = to_be_processed[0]
            new_graph = {}
            to_be_processed = self.__add_node_to_graph(node, new_graph, to_be_processed)
            graphs.append(new_graph)

        step_size = .5
        prev_width = 0
        for graph in graphs:
            if prev_width > 0:
                graph_offset = prev_width + step_size
            else:
                graph_offset = 0

            width = max([len(x) for x in graph.items()]) * step_size
            for key in graph.keys():
                row_width = len(graph[key]) * step_size
                diff = width - row_width
                for i, n in enumerate(graph[key]):
                    x = .5 * diff + i * step_size + graph_offset
                    n.coordinates = (x, key)

            prev_width = graph_offset + width

    def __add_node_to_graph(self, node, graph, to_be_processed):
        """Adds a node to a dictionary graph object. Used by rearrange_nodes(). When run the node is removed from the to_be_processed list.

        :param node: The node to add to the graph.
        :param graph: A dictionary object with coordinates as key, and a list of nodes as value.
        :param to_be_processed: A list of nodes that still have to be processed.
        :return: A list containing the nodes which still have to be processed by this algorithm.
        """
        to_be_processed = np.delete(to_be_processed, np.where(to_be_processed == node))
        if node.coordinates[1] in graph.keys():
            graph[node.coordinates[1]].append(node)
        else:
            graph[node.coordinates[1]] = [node]

            for neighbour in node.neighbours:
                if neighbour in to_be_processed:
                    to_be_processed = self.__add_node_to_graph(neighbour, graph, to_be_processed)
        return to_be_processed

