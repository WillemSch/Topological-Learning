import torch


def add_tuples(a, b):
    """Method to add 2 tuples together. If lengths of tuples mismatch the last n indices are of the longest tuple are
        unaffected.

    :param a: Tuple of length i.
    :param b: Tuple of length j.
    :return: Tuple of length max(i,j).
    """
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
    """Multiply the contents of a tuple by a scalar.

    :param scalar: The scalar to mupltiply the tuple by.
    :param tup: The tuple to scale.
    :return: A scaled tuple.
    """
    return tuple([scalar * x for x in tup])


def average_tuples(a, b):
    """Get the element-wise average of 2 tuples.

    :param a: Tuple of length i
    :param b: Tuple of length i
    :return: Tuple of length i
    """
    assert len(a) == len(b)
    return tuple([sum(x)/2 for x in zip(a, b)])


def create_distance_matrix(points):
    """Creates a distance matrix of a set of points, using the norm distance.

    :param points: A list or pytorch Tensor of coordinates of points.
    :return: A pytorch Tensor containing the distance matrix.
    """
    if not isinstance(points, torch.Tensor):
        points = torch.tensor(points)
    dist = torch.zeros((len(points), len(points)))
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dist[i, j] = dist[j, i] = torch.linalg.norm(points[i] - points[j])
    return dist
