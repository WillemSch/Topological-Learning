import math
import numpy as np
from scipy.special import comb
from itertools import combinations


# ================================Column reduction===========================================
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


# ================================Rips complex + HBDSCAN===========================================
def rips(distance_matrix, dimensions=2, max_radius=np.inf, step_size=.1):
    radius = 0
    filtered_simplexes = np.array([np.zeros(len(distance_matrix[0])) for _ in range(distance_matrix.shape[0])])
    labels = [0. for _ in range(distance_matrix.shape[0])]

    while not __all_simplexes_found(distance_matrix.shape[0], filtered_simplexes, dimensions) and radius <= max_radius:
        for i in range(distance_matrix.shape[0]):
            for j in range(i + 1, distance_matrix.shape[0]):
                new_simplexes = [[i, j]]

                while len(new_simplexes) > 0:  # Recursively add simplexes if a higher-dimensional simplex is created
                    next_simplexes = []
                    for simplex in new_simplexes:
                        if distance_matrix[i][j] < 2 * radius and not __simplex_exists(simplex, filtered_simplexes):
                            filtered_simplexes = __add_complex(simplex, filtered_simplexes)
                            labels.append(radius)

                            # If the new simplex adds higher dimensional simplexes add those as well
                            # and again check if higher dimensional simplexes are made
                            next_simplexes += __new_complexes_created(simplex, filtered_simplexes, dimensions)
                    new_simplexes = next_simplexes

        radius += step_size

    return filtered_simplexes, labels


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

        # possible optimisation: filter for simplexes that are within radius
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


def hbdscan_rips(distance_matrix, max_dimensions=2, max_radius=np.inf, step_size=.1, k_core=5):
    altered_distances = np.zeros(distance_matrix.shape)
    for i in range(distance_matrix.shape[0]):
        for j in range(i + 1, distance_matrix.shape[0]):
            i_core = __core_distance(distance_matrix, i, k_core)
            j_core = __core_distance(distance_matrix, j, k_core)
            altered_distances[i][j] = altered_distances[j][i] = max(i_core, j_core, distance_matrix[i][j])
    return rips(altered_distances, dimensions=max_dimensions, max_radius=max_radius, step_size=step_size)


def __core_distance(distance_matrix, origin_index, k):
    return np.max(np.sort(distance_matrix[origin_index])[1:k + 1])


# ================================Persistence Image===========================================
def filtered_complexes_to_tuples(filtered_complexes, labels):
    tuples = []
    used = []
    reduced = reduce_columns(filtered_complexes)
    for i, col in enumerate(reduced.T):
        if np.count_nonzero(col) > 0:
            death = labels[i]
            j = np.max(np.where(col == 1))
            birth = labels[j]
            used.append(i)
            used.append(j)
            tuples.append([birth, death])

    tuples = np.array(tuples)
    for i in [x for x in range(reduced.shape[0]) if x not in used]:
        # Empty rows mean a simplex with death value infinity
        tuples = np.append(tuples, [[labels[i], np.inf]], axis=0)
    return tuples


def transform_to_birth_persistence(birth_death_tuples, infinity_replacement):
    birth_persistence_tuples = []
    for (birth, death) in birth_death_tuples:
        if death == np.inf:
            persistence = infinity_replacement
        else:
            persistence = death - birth
        birth_persistence_tuples.append((birth, persistence))
    return np.array(birth_persistence_tuples)


class PersistenceImage:
    def __init__(self, persistence_matrix, labels):
        self.max_weight = 1

        tuples = filtered_complexes_to_tuples(persistence_matrix, labels)
        # Replace death = infinity with 2x the greatest death value
        masked_tuples = np.ma.array(tuples, mask=~np.isfinite(tuples)).flatten()
        max_idx = np.ma.argmax(masked_tuples, fill_value=-np.inf)
        self.points = transform_to_birth_persistence(tuples, infinity_replacement=2 * tuples.flatten()[max_idx])

    def transform(self, resolution, x_min=None, x_max=None, y_min=None, y_max=None, kernel_spread=2):
        if x_min is None:
            x_min = 0
        if x_max is None:
            x_max = np.max(self.points.T[0]) + .5
        if y_min is None:
            y_min = 0
        if y_max is None:
            y_max = np.max(self.points.T[1]) + .5

        pixels = self.__to_pixels(resolution, x_min, x_max, y_min, y_max, kernel_spread)
        return pixels

    def __to_pixels(self, resolution, x_min, x_max, y_min, y_max, kernel_spread):
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_step = x_range / resolution
        y_step = y_range / resolution
        x_centering_offset = .5 * x_step
        y_centering_offset = .5 * y_step
        pixels = np.zeros((resolution, resolution))

        for i in range(0, resolution):
            x = x_centering_offset + i * x_step
            for j in range(0, resolution):
                y = y_centering_offset + j * y_step
                pixels[i, j] = self.__density_at_coord(x, y, y_max, kernel_spread)

        return pixels

    def __density_at_coord(self, x, y, max_y, kernel_spread):
        density = 0

        for point in self.points:
            # Linear weight function
            point_weight = self.max_weight * point[1] / max_y
            density += point_weight * self.gaussian_prob(point, kernel_spread, x, y)

        return density

    def gaussian_prob(self, point, kernel_spread, x, y):
        return 1/(2 * np.pi * kernel_spread**2) * np.exp(
            -1 * (((x - point[0])**2 + (y - point[1])**2)/(2 * kernel_spread**2))
        )


# ================================Persistence Landscape===========================================
def transform_to_mid_half_life(birth_death_tuples, infinity_replacement):
    mid_half_life_tuples = []
    for (birth, death) in birth_death_tuples:
        if death == np.inf:
            death = infinity_replacement
        half_life = (death - birth) / 2
        mid_life = (birth + death) / 2
        mid_half_life_tuples.append((mid_life, half_life))
    return np.array(mid_half_life_tuples)


class PersistenceLandscape:
    def __init__(self, persistence_matrix, labels):
        tuples = filtered_complexes_to_tuples(persistence_matrix, labels)
        # Replace death = infinity with 2x the greatest death value
        masked_tuples = np.ma.array(tuples, mask=~np.isfinite(tuples)).flatten()
        max_idx = np.ma.argmax(masked_tuples, fill_value=-np.inf)
        self.points = transform_to_mid_half_life(tuples, infinity_replacement=2 * tuples.flatten()[max_idx])

    def transform(self, steps, x_min=None, x_max=None):
        if x_min is None:
            x_min = min(self.points, key=lambda x: x[0])[0]
        if x_max is None:
            x_max = max(self.points, key=lambda x: x[0])[0]

        result = np.zeros((len(self.points), steps))

        step_size = (x_max - x_min) / steps
        for i in range(steps):
            x = step_size * i + x_min
            for y, point in enumerate(self.points):
                result[y, i] = self.__get_triangle_height(x, point)

        result = np.flip(np.sort(result, axis=0), axis=0)

        return (x_min, x_max), result

    def __get_triangle_height(self, x, peak):
        distance = math.fabs(peak[0] - x)
        return max(0, peak[1] - distance)

