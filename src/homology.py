import numpy as np
from scipy.special import comb
from itertools import combinations


def reduce_columns(filtered_complex_matrix):
    # For easier manipulation of rows, transpose the matrix:
    filtered_complex_matrix = filtered_complex_matrix.T
    for j in range(1, len(filtered_complex_matrix)):
        j_prime = __find_prev_row_same_low(j, filtered_complex_matrix)
        while j_prime is not None:
            filtered_complex_matrix[j] = (filtered_complex_matrix[j] + filtered_complex_matrix[j_prime]) % 2
            j_prime = __find_prev_row_same_low(j, filtered_complex_matrix)
    return filtered_complex_matrix.T


def __find_prev_row_same_low(j, filtered_complex_matrix):
    for i in range(0, j):
        if np.max(filtered_complex_matrix[i]) > 0 and np.max(filtered_complex_matrix[j]) > 0 \
                and __low(filtered_complex_matrix[i]) == __low(filtered_complex_matrix[j]):
            return i
    return None


def __low(row):
    indices = np.argwhere(row == 1)
    return indices[-1]


def ribs(distance_matrix, dimensions=2, max_radius=np.inf):
    radius = 0
    step_size = .1
    filtered_simplexes = np.array([np.zeros(len(distance_matrix[0])) for x in range(distance_matrix.shape[0])])

    while not __all_simplexes_found(distance_matrix.shape[0], filtered_simplexes, dimensions) and radius <= max_radius:
        for i in range(distance_matrix.shape[0]):
            for j in range(i + 1, distance_matrix.shape[0]):
                new_simplexes = [[i, j]]

                while len(new_simplexes) > 0:  # Recursively add simplexes if a higher-dimensional simplex is created
                    next_simplexes = []
                    for simplex in new_simplexes:
                        if distance_matrix[i][j] < 2 * radius and not __simplex_exists(simplex, filtered_simplexes):
                            filtered_simplexes = __add_complex(simplex, filtered_simplexes)

                            # If the new simplex adds higher dimensional simplexes add those as well
                            # and again check if higher dimensional simplexes are made
                            next_simplexes += __new_complexes_created(simplex, filtered_simplexes, dimensions)
                    new_simplexes = next_simplexes

        radius += step_size

    return filtered_simplexes


def __add_complex(new_complex, filtered_simplexes):
    filtered_simplexes = np.concatenate((filtered_simplexes, np.zeros((1, filtered_simplexes.shape[1]))), axis=0)
    new_col = np.zeros((filtered_simplexes.shape[0], 1))
    for x in new_complex:
        new_col[x] = 1.
    filtered_simplexes = np.concatenate((filtered_simplexes, new_col), axis=1)
    return filtered_simplexes


def __new_complexes_created(added_complex, filtered_simplexes, max_dimensions):
    if len(added_complex) >= max_dimensions + 1:
        return []
    else:
        new_complexes = []
        dim = len(added_complex)
        tuples = __to_index_tuples(filtered_simplexes, dim)
        if tuples.shape[0] > dim:
            for group in combinations(tuples, dim + 1):
                group = np.array(group)
                # the first column of the group contains the indices of the simplexes, so we extract and remove these
                indices = group[:, 0]
                group = group[:, 1:]
                uniques = np.unique(group)
                if uniques.shape[0] == dim + 1:
                    new_complexes.append(indices)
        return new_complexes


def __simplex_exists(simplex, filtered_simplexes):
    for col in filtered_simplexes.T:
        match = True
        for index, value in enumerate(col):
            if (index in simplex and value == 0.) or (index not in simplex and value == 1.):
                match = False
        if match:
            return True
    return False


def __all_simplexes_found(zero_simplex_count, filtered_simplexes, max_dimensions):
    max_c = 0
    for d in range(max_dimensions + 1):
        max_c += comb(zero_simplex_count, d + 1)
    return max_c == filtered_simplexes.shape[0]


def __to_index_tuples(filtered_simplexes, dimension_filter=None):
    simplexes = []
    for index, col in enumerate(filtered_simplexes.T):
        simplex = np.where(col == 1.)
        if dimension_filter is not None:
            if len(simplex[0]) == dimension_filter:
                simplexes.append(np.insert(simplex[0], 0, index))
        else:
            if len(simplex[0]) > 0:
                simplexes.append(np.insert(simplex[0], 0, index))
    return np.array(simplexes)
