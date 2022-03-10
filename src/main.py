import graph
from sklearn import datasets
import som as SOM
import gng as GNG
import reeb as Reeb
import numpy as np
import homology as hm

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


def test_reeb():
    x, _ = datasets.make_blobs(random_state=47, center_box=(-5, 5), centers=7)
    reeb = Reeb.Reeb()
    g = reeb.map(x, 7, overlap=0.2)


def test_column_reduction():
    x = np.array([
        [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
    print(x)
    print(hm.reduce_columns(x))


def test_ribs():
    x = np.array([
        [0, 5, 7, 4],
        [5, 0, 3, 7],
        [7, 3, 0, 11],
        [4, 7, 11, 0]
    ])
    # print(hm.rips(x, 3, 7))
    # print(hm.rips(x, 1, 6))
    print(hm.hbdscan_rips(x, k_core=5))
    print(hm.hbdscan_rips(x, k_core=2))


def test_persistance_image():
    x = np.array([
        [0, 5, 7, 4],
        [5, 0, 3, 7],
        [7, 3, 0, 11],
        [4, 7, 11, 0]
    ])
    filtered_complexes, labels = hm.rips(x, 2, 7)
    print(hm.filtered_complexes_to_tuples(hm.reduce_columns(filtered_complexes), labels))
    per_img = hm.PersistenceImage(filtered_complexes, labels)
    print(per_img.transform(resolution=5).T)

# Debug tests
if __name__ == '__main__':
    # x = graph.create_grid((4, 4))
    # x.to_nx_graph()
    # print(x.get_edges())
    # print(x.nodes[(0, 0)].get_nth_neighbours(9))
    # test_lines()
    # test_gng()
    # test_reeb()
    # test_column_reduction()
    # test_ribs()
    test_persistance_image()
