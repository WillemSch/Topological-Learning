def add_tuples(a, b):
    len(a)
    len(b)
    if len(a) == len(b):
        return tuple([sum(x) for x in zip(a,b)])
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
