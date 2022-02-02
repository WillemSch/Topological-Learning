import Graph
from sklearn import datasets
import SelfOrganisingGraphs


def test_lines():
    lines = Graph.Graph(7, dimensions=2)

    lines.nodes[1].connect(lines.nodes[0])
    lines.nodes[1].connect(lines.nodes[2])
    lines.nodes[4].connect(lines.nodes[3])
    lines.nodes[4].connect(lines.nodes[5])
    lines.nodes[5].connect(lines.nodes[6])

    # print(lines.nodes[1].get_nth_neighbours(9))

    x, y = datasets.make_blobs(random_state=47, center_box=(-5,5))

    som = SelfOrganisingGraphs.SOM(lines)
    som.fit(x)


# Debug tests
if __name__ == '__main__':
    x = Graph.create_grid((4,4))
    x.to_nx_graph()
    # print(x.get_edges())
    # print(x.nodes[(0, 0)].get_nth_neighbours(9))
    test_lines()
