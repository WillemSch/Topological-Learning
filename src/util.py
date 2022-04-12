import torch


def add_tuples(a, b):
    len(a)
    len(b)
    if len(a) == len(b):
        return tuple([sum(x) for x in zip(a, b)])
    else:
        if len(a) > len(b):
            longest = a
            difference = len(a) - len(b)
        else:
            longest = b
            difference = len(b) - len(a)

        res = [sum(x) for x in zip(a, b)] + list(longest)[-difference:]
        return tuple(res)


def multiply_tuple(scalar, tup):
    return tuple([scalar * x for x in tup])


def average_tuples(a, b):
    assert len(a) == len(b)
    return tuple([sum(x)/2 for x in zip(a, b)])


def create_distance_matrix(points):
    if not isinstance(points, torch.Tensor):
        points = torch.tensor(points)
    dist = torch.zeros((len(points), len(points)))
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dist[i, j] = dist[j, i] = torch.linalg.norm(points[i] - points[j])
    return dist
