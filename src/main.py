import graph
from sklearn import datasets
import som as SOM
import gng as GNG


def test_lines():
    lines = graph.Graph(7, dimensions=2)

    lines.nodes[1].connect(lines.nodes[0])
    lines.nodes[1].connect(lines.nodes[2])
    lines.nodes[4].connect(lines.nodes[3])
    lines.nodes[4].connect(lines.nodes[5])
    lines.nodes[5].connect(lines.nodes[6])

    # print(lines.nodes[1].get_nth_neighbours(9))

    x, y = datasets.make_blobs(random_state=47, center_box=(-5,5))

    som = SOM.SOM(lines)
    som.fit(x)


def test_gng():
    x, _ = datasets.make_blobs(random_state=47, center_box=(-5, 5), centers=7)
    gng = GNG.GNG(x)
    gng.train(100, 0.1, 100, 50, .5, .2)
    gng.graph.to_nx_graph()


# Debug tests
if __name__ == '__main__':
    # x = graph.create_grid((4, 4))
    # x.to_nx_graph()
    # print(x.get_edges())
    # print(x.nodes[(0, 0)].get_nth_neighbours(9))
    # test_lines()
    test_gng()
