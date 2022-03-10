import numpy
import graph as Graph
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np


def __get_preimage_for_interval(data, processed_data, interval):
    return np.array([data[i] for i, x in enumerate(processed_data) if interval[0] <= x[0] <= interval[1]])


def __has_overlapping_data(a_data, b_data, threshold=1):
    return len(numpy.intersect1d(a_data.flatten(), b_data.flatten())) >= threshold


def first_dimension(x):
    return x[0]


def second_dimension(x):
    return x[1]


class Reeb:
    # function has to map data to a real number
    def __init__(self, function=second_dimension):
        self.function = function
        self.g = Graph.Graph(0, dimensions=2)

    def map(self, data, intervals, overlap=.2, max_k=5):
        processed_data = [[self.function(x)] for x in data]
        data_range = (np.min(processed_data), np.max(processed_data))
        range_distance = np.abs(data_range[1] - data_range[0])
        interval_size = (range_distance * (1 + overlap)) / intervals
        node_distance = range_distance * .2

        interval_end = interval_size * overlap + data_range[0]
        prev_nodes = []
        for i in range(intervals):
            interval_start = interval_end - interval_size * overlap
            interval_end = interval_start + interval_size
            interval = (interval_start, interval_end)
            preimage_data = __get_preimage_for_interval(data, processed_data, interval)

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
                    if __has_overlapping_data(node_data, old[1]):
                        new_node.connect(old[0])
                new_nodes.append((new_node, node_data))
            prev_nodes = new_nodes

        self.rearrange_nodes()
        return self.g

    def rearrange_nodes(self):
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
        to_be_processed = np.delete(to_be_processed, np.where(to_be_processed == node))
        if node.coordinates[1] in graph.keys():
            graph[node.coordinates[1]].append(node)
        else:
            graph[node.coordinates[1]] = [node]

            for neighbour in node.neighbours:
                if neighbour in to_be_processed:
                    to_be_processed = self.__add_node_to_graph(neighbour, graph, to_be_processed)
        return to_be_processed

