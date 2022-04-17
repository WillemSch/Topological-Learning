import math
import numpy as np
from scipy.special import comb
from itertools import combinations


# ================================Column reduction===========================================
def reduce_columns(filtered_complex_matrix):
    """Applies column reduction over a filtered complex matrix.

    :param filtered_complex_matrix: A filtered complex matrix. (2 dimensional numpy array)
    :return: A column-reduced filtered complex matrix
    """
    # For easier manipulation of columns, transpose the matrix:
    filtered_complex_matrix = filtered_complex_matrix.T
    for j in range(1, len(filtered_complex_matrix)):
        j_prime = __find_prev_row_same_low(j, filtered_complex_matrix)
        while j_prime is not None:
            filtered_complex_matrix[j] = (filtered_complex_matrix[j] + filtered_complex_matrix[j_prime]) % 2
            j_prime = __find_prev_row_same_low(j, filtered_complex_matrix)
    return filtered_complex_matrix.T  # Transpose again before returning


def __find_prev_row_same_low(j, filtered_complex_matrix):
    """Used in the reduce_columns function. Find a previous row that has its last '1'-entry on the same index as the row
        at the given index.

    :param j: The index of the row to compare to.
    :param filtered_complex_matrix: The filtered complex matrix
    :return: The index of a previous row that has its last '1'-entry on the same index, None if such a row does not exist.
    """
    for i in range(0, j):
        if np.max(filtered_complex_matrix[i]) > 0 and np.max(filtered_complex_matrix[j]) > 0 \
                and __low(filtered_complex_matrix[i]) == __low(filtered_complex_matrix[j]):
            return i
    return None


def __low(row):
    """Find the index of the last '1' in a row.

    :param row: A numpy array.
    :return: The index of the last '1' in the row.
    """
    indices = np.argwhere(row == 1)
    return indices[-1]


# ================================Rips complex + HBDSCAN===========================================
def rips(distance_matrix, dimensions=2, max_radius=np.inf, step_size=.1):
    """Applies the rips algorithm over a given distance matrix to produce a filtered simplicial complex. This algorithm
        terminates either when all simplexes are found, or when the maximum radius is reached.

    :param distance_matrix: A distance matrix of a set of points
    :param dimensions: Optional, default 2 - The upper bound of dimensions for simplexes created.
    :param max_radius: Optional, default np.inf (infinity) - The maximum
    :param step_size: Optional, default 0.1 - The amount with which the radius is increased at each step.
    :return: A filtered complex matrix, and a list with the corresponding birth radii.
    """
    radius = 0
    # We start with all 0-simplexes with birth value 0.
    filtered_simplexes = np.array([np.zeros(len(distance_matrix[0])) for _ in range(distance_matrix.shape[0])])
    labels = [0. for _ in range(distance_matrix.shape[0])]

    while not __all_simplexes_found(distance_matrix.shape[0], filtered_simplexes, dimensions) and radius <= max_radius:

        for i in range(distance_matrix.shape[0]):
            for j in range(i + 1, distance_matrix.shape[0]):
                new_simplexes = [[i, j]]  # Create edge

                while len(new_simplexes) > 0:  # Recursively add simplexes if a higher-dimensional simplex is created
                    next_simplexes = []
                    for simplex in new_simplexes:
                        if distance_matrix[i][j] < 2 * radius and not __simplex_exists(simplex, filtered_simplexes):
                            filtered_simplexes = __add_complex(simplex, filtered_simplexes)
                            labels.append(radius)  # Save the birth label

                            # If the new simplex adds higher dimensional simplexes add those as well
                            # and again check if higher dimensional simplexes are made
                            next_simplexes += __new_complexes_created(simplex, filtered_simplexes, dimensions)
                    new_simplexes = next_simplexes

        radius += step_size

    return filtered_simplexes, labels


def __add_complex(new_complex, filtered_simplexes):
    """Adds a simplex to the filtered simplex matrix, and expands the matrix to fit the new number of simplexes

    :param new_complex: A list of indices of simplexes that created this simplex.
    :param filtered_simplexes: The filtered simplex matrix.
    :return: An expanded filtered simplex matrix, with the new simplex added to it.
    """
    filtered_simplexes = np.concatenate((filtered_simplexes, np.zeros((1, filtered_simplexes.shape[1]))), axis=0)
    new_col = np.zeros((filtered_simplexes.shape[0], 1))
    for x in new_complex:
        new_col[x] = 1.
    filtered_simplexes = np.concatenate((filtered_simplexes, new_col), axis=1)
    return filtered_simplexes


def __new_complexes_created(added_complex, filtered_simplexes, max_dimensions):
    """Checks if complexes are created because of the birth of a simplex.

    :param added_complex: The newly added complex; list of indices of simplexes that created it.
    :param filtered_simplexes: The filtered simplex matrix.
    :param max_dimensions: The upper bound of dimensions for complexes.
    :return: A list of complexes that are created by the birth of added_complex; A list of lists of indices.
    """
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
    """Checks whether a given simplex exists in a filtered simplex matrix.

    :param simplex: A list of indices that created the simplex.
    :param filtered_simplexes: A filtered simplex matrix.
    :return: True if the simplex exists in the matrix, False if not.
    """
    for col in filtered_simplexes.T:
        match = True
        for index, value in enumerate(col):
            if (index in simplex and value == 0.) or (index not in simplex and value == 1.):
                match = False
        if match:
            return True
    return False


def __all_simplexes_found(zero_simplex_count, filtered_simplexes, max_dimensions):
    """Checks whether all simplexes are fount in a filtered simplex matrix, given an upper bound on dimensionality of
        simplexes.

    :param zero_simplex_count: The amount of 0-simplexes.
    :param filtered_simplexes: The filtered simplex matrix.
    :param max_dimensions: The upper bound on dimensionality for simplexes.
    :return: True the amount of possible simplexes equals the length of the filtered simplex matrix, False otherwise.
    """
    max_c = 0
    for d in range(max_dimensions + 1):
        max_c += comb(zero_simplex_count, d + 1)
    return max_c == filtered_simplexes.shape[0]


def __to_index_tuples(filtered_simplexes, dimension_filter=None):
    """Get all simplexes from a filtered simplex matrix as tuples of indices of the simplexes that created them.

    :param filtered_simplexes: A filtered simplex matrix.
    :param dimension_filter: Optional, default None - An optional filter to only select simplexes of a given
        dimensionality
    :return: A list of lists of indices
    """
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
    """A simple wrapper function that applies HBDSCAN over a distance matrix before passing it through the rips()
        function.

    :param distance_matrix: A distance matrix of a set of points.
    :param max_dimensions: Optional, default 2 - The upper bound of dimensions for simplexes created.
    :param max_radius: Optional, default np.inf (infinity) - The maximum
    :param step_size: Optional, default 0.1 - The amount with which the radius is increased at each step.
    :param k_core: The neighbourhood size (K) to be used to calculate the core distances.
    :return: A filtered complex matrix, and a list with the corresponding birth radii.
    """
    altered_distances = np.zeros(distance_matrix.shape)
    for i in range(distance_matrix.shape[0]):
        for j in range(i + 1, distance_matrix.shape[0]):
            i_core = __core_distance(distance_matrix, i, k_core)
            j_core = __core_distance(distance_matrix, j, k_core)
            altered_distances[i][j] = altered_distances[j][i] = max(i_core, j_core, distance_matrix[i][j])
    return rips(altered_distances, dimensions=max_dimensions, max_radius=max_radius, step_size=step_size)


def __core_distance(distance_matrix, origin_index, k):
    """Calculates the core distance for a given point in a distance matrix.

    :param distance_matrix: A distance matrix of a set of points.
    :param origin_index: The index of the point to calculate the core distance for.
    :param k: The neighbourhood size.
    :return: The distance between this point and its Kth nearest neighbour.
    """
    return np.max(np.sort(distance_matrix[origin_index])[1:k + 1])


# ================================Persistence Image===========================================
def filtered_complexes_to_tuples(filtered_complexes, labels):
    """Generate a list of [birth,death] tuples from a filtered complex matrix, and it's corresponding birth-labels.

    :param filtered_complexes: A filtered complex matrix.
    :param labels: The birth-labels corresponding to the filtered complex matrix.
    :return: A list of [birth, death] tuples.
    """

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
        tuples = np.append(tuples, [[labels[i], np.inf]], axis=0)
    return tuples


def transform_to_birth_persistence(birth_death_tuples, infinity_replacement):
    """Transform a list of [birth, death] tuples to a list of [birth, persistence] tuples.

    :param birth_death_tuples: A list of [birth, death] tuples.
    :param infinity_replacement: If persistence of a simplex is infinity its persistence will be replaced with this
        value.
    :return: A list of [birth, persistence] tuples.
    """

    birth_persistence_tuples = []
    for (birth, death) in birth_death_tuples:
        if death == np.inf:
            persistence = infinity_replacement
        else:
            persistence = death - birth
        birth_persistence_tuples.append((birth, persistence))
    return np.array(birth_persistence_tuples)


class PersistenceImage:
    """A class to produce Persistence Images from a filtered complex matrix.

    :param filtered_complexes: A filtered complex matrix.
    :param labels: The birth-labels corresponding to the filtered complex matrix.
    """

    def __init__(self, filtered_complexes, labels):
        """Initializes the PersistenceImage class. Pre-processes the filtered complex matrix and labels so a persistence
            image can be generated.

        :param filtered_complexes: A filtered complex matrix.
        :param labels: The birth-labels corresponding to the filtered complex matrix.
        """
        self.max_weight = 1

        tuples = filtered_complexes_to_tuples(filtered_complexes, labels)
        # Replace death = infinity with 2x the greatest death value
        masked_tuples = np.ma.array(tuples, mask=~np.isfinite(tuples)).flatten()
        max_idx = np.ma.argmax(masked_tuples, fill_value=-np.inf)  # <- The index of the simplex with highest death value that isn't infinity
        self.points = transform_to_birth_persistence(tuples, infinity_replacement=2 * tuples.flatten()[max_idx])

    def transform(self, resolution, x_min=None, x_max=None, y_min=None, y_max=None, kernel_spread=2):
        """Generates a persistence image with a given resolution for the dataset.

        :param resolution: The size of the resulting matrix (Integer); result will be matrix of shape (resolution,
            resolution).
        :param x_min: Optional, default None - Lower bound for the x-axis, if not defined: 0.
        :param x_max: Optional, default None - Upper bound for the x-axis, if not defined: Maximum x value + 0.5.
        :param y_min: Optional, default None - Lower bound for the y-axis, if not defined: 0.
        :param y_max: Optional, default None - Upper bound for the y-axis, if not defined: Maximum y value + 0.5.
        :param kernel_spread: Optional, default 2 - Defines the kernel-spread used to calculate density.
        :return: A matrix of shape (resolution, resolution) with describing the density at each pixel.
        """
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
        """Used by transform(). Generate the 'pixels' for the dataset.

        :param resolution: The size of the resulting matrix (Integer); result will be matrix of shape (resolution,
            resolution).
        :param x_min: Lower bound for the x-axis.
        :param x_max: Upper bound for the x-axis.
        :param y_min: Lower bound for the y-axis.
        :param y_max: Upper bound for the y-axis.
        :param kernel_spread: Defines the kernel-spread used to calculate density.
        :return: A matrix of shape (resolution, resolution) with describing the density at each pixel.
        """
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
                pixels[j, i] = self.__density_at_coord(x, y, y_max, kernel_spread)

        return pixels

    def __density_at_coord(self, x, y, max_y, kernel_spread):
        """Calculates the Gaussian density of the dataset at a given coordinate.

        :param x: X coordinate.
        :param y: Y coordinate.
        :param max_y: The upper_bound of the image (used for the linear weighting function).
        :param kernel_spread: Defines the kernel-spread used to calculate density.
        :return: The density at the given coordinate.
        """
        density = 0

        for point in self.points:
            # Linear weight function
            point_weight = self.max_weight * point[1] / max_y
            density += point_weight * self.gaussian_prob(point, kernel_spread, x, y)

        return density

    def gaussian_prob(self, point, kernel_spread, x, y):
        """Calculates the Gaussian probability at a given (x,y) coordinate from a kernel at a given point.

        :param point: The center of the kernel.
        :param kernel_spread: Defines the kernel-spread used to calculate density.
        :param x: X coordinate.
        :param y: Y coordinate.
        :return: The gaussian probability at the given coordinate.
        """
        return 1/(2 * np.pi * kernel_spread**2) * np.exp(
            -1 * (((x - point[0])**2 + (y - point[1])**2)/(2 * kernel_spread**2))
        )


# ================================Persistence Landscape===========================================
def transform_to_mid_half_life(birth_death_tuples, infinity_replacement):
    """Transform a list of [birth, death] tuples to a list of [mid-life, half-life] tuples.

    :param birth_death_tuples: A list of [birth, death] tuples.
    :param infinity_replacement: If death of a simplex is infinity its death will be replaced with this value.
    :return: A list of [mid-life, half-life] tuples.
    """
    mid_half_life_tuples = []
    for (birth, death) in birth_death_tuples:
        if death == np.inf:
            death = infinity_replacement
        half_life = (death - birth) / 2
        mid_life = (birth + death) / 2
        mid_half_life_tuples.append((mid_life, half_life))
    return np.array(mid_half_life_tuples)


class PersistenceLandscape:
    """A class to produce Persistence Images from a filtered complex matrix.

    :param filtered_complexes: A filtered complex matrix.
    :param labels: The birth-labels corresponding to the filtered complex matrix.
    """

    def __init__(self, filtered_complexes, labels):
        """Initializes the PersistenceLandscape class. Pre-processes the filtered complex matrix and labels so a
            persistence landscape can be generated.

        :param filtered_complexes: A filtered complex matrix.
        :param labels: The birth-labels corresponding to the filtered complex matrix.
        """
        tuples = filtered_complexes_to_tuples(filtered_complexes, labels)
        # Replace death = infinity with 2x the greatest death value
        masked_tuples = np.ma.array(tuples, mask=~np.isfinite(tuples)).flatten()
        max_idx = np.ma.argmax(masked_tuples, fill_value=-np.inf)
        self.points = transform_to_mid_half_life(tuples, infinity_replacement=2 * tuples.flatten()[max_idx])

    def transform(self, steps, x_min=None, x_max=None):
        """Generate a persistence landscape at a given resolution for the dataset.

        :param steps: The amount of steps in the discretization step.
        :param x_min: Optional, default None - The lower bound of the x-axis, if not specified the lowest x of the
            dataset is used.
        :param x_max: Optional, default None - The upper bound of the x-axis, if not specified the highest x of the
            dataset is used.
        :return: A 2 dimensional numpy array with the persistence landscape functions.
        """
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
        """Get the height of a triangle at a given point on the x axis.

        :param x: A point on the x-axis.
        :param peak: A tuple of (x, y) coordinates.
        :return: The height of the triangle with a given peak at x. Triangles have slope 1.
        """
        # Simply subtract x-axis distance from the height, since all triangles have slope 1
        distance = math.fabs(peak[0] - x)
        return max(0, peak[1] - distance)

