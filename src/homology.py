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
    filtered_complex_matrix = np.copy(filtered_complex_matrix).T
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


def get_creator_simplex_indices(filtered_complex, dimension_filter=None):
    """For all n-simplexes, where n > 0, return the 0-simplex with the highest birth filtration value that makes up the
    original simplex.

    :param filtered_complex: A non-column reduced filtered simplicial complex (numpy nd-array)
    :param dimension_filter: Optional, default None - List of Integers, only return creator simplexes for n-simplexes
        where n in ```dimension_filter```
    :return: A list of indices of creator 0-simplexes.
    """
    indices = set()
    for col in filtered_complex.T:
        non_zero_indices = np.where(col != 0)[0]
        if len(non_zero_indices) == 0:
            continue

        if dimension_filter is not None and len(non_zero_indices) - 1 in dimension_filter:
            prev_non_zero_indices = non_zero_indices
            simplex_col = filtered_complex.T[np.max(non_zero_indices)]
            simplex_non_zero_indices = np.where(simplex_col != 0)[0]

            while simplex_non_zero_indices.shape[0] > 0:
                prev_non_zero_indices = simplex_non_zero_indices
                simplex_col = filtered_complex.T[np.max(simplex_non_zero_indices)]
                simplex_non_zero_indices = np.where(simplex_col != 0)[0]

            indices.add(np.max(prev_non_zero_indices))

    return list(indices)


# ================================Rips complex + HBDSCAN===========================================
class Rips:
    """A class to apply the rips algorithm to a distance matrix.

    :param dimensions: Optional, default 2 - The upper bound of dimensions for simplexes created.
    :param max_radius: Optional, default np.inf (infinity) - The maximum radius of spheres in the Rips algorithm.
    """

    def __init__(self, dimensions=2, max_radius=np.inf):
        """Initialize the Rips class

        :param dimensions: Optional, default 2 - The upper bound of dimensions for simplexes created.
        :param max_radius: Optional, default np.inf (infinity) - The maximum radius of spheres in the Rips algorithm.
        """
        self.dimensions = dimensions
        self.max_radius = max_radius
        self.filtered_simplexes = None
        self.labels = None

    def fit(self, distance_matrix):
        """Fit the Rips class to the distance matrix.

        :param distance_matrix: A distance matrix of a set of points.
        :return: This class instance.
        """
        self.filtered_simplexes = np.array([np.zeros(len(distance_matrix[0])) for _ in range(distance_matrix.shape[0])])
        self.labels = [0. for _ in range(distance_matrix.shape[0])]
        return self


    def transform(self, distance_matrix):
        """Applies the Rips algorithm over a given distance matrix to produce a filtered simplicial complex. This
        algorithm terminates either when all simplexes are found, or when the maximum radius is reached.

        :param distance_matrix: A distance matrix of a set of points
        :return: A filtered complex matrix, and a list with the corresponding birth radii.
        """
        distance_matrix = distance_matrix.copy()

        for j in range(distance_matrix.shape[0]):
            # Set all nodes to have infinity distance to themselves, so they don't get added again.
            distance_matrix[j, j] = np.inf

        min_index = np.unravel_index(np.argmin(distance_matrix, axis=None), distance_matrix.shape)

        while distance_matrix[min_index] != np.inf and distance_matrix[min_index] <= self.max_radius * 2:
            new_complexes = [list(min_index)]
            while len(new_complexes) > 0:
                next_complexes = []
                for complex in new_complexes:
                    self.filtered_simplexes = add_complex(complex, self.filtered_simplexes)
                    self.labels.append(distance_matrix[min_index] / 2)
                    next_complexes += new_complexes_created(complex, self.filtered_simplexes, max_dimensions=self.dimensions)
                new_complexes = next_complexes

            distance_matrix[min_index] = np.inf
            distance_matrix[(min_index[1], min_index[0])] = np.inf
            min_index = np.unravel_index(np.argmin(distance_matrix, axis=None), distance_matrix.shape)

        return self.filtered_simplexes, self.labels


def add_complex(new_complex, filtered_simplexes):
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


def new_complexes_created(added_complex, filtered_simplexes, max_dimensions):
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
        tuples = to_index_tuples(filtered_simplexes, dim)
        added_complex_index = len(filtered_simplexes) - 1

        # We only care about tuples that share at least 1 origin simplex with the added_complex
        # (meaning they are connected)
        filtered_tuples = [x for x in tuples if len(np.intersect1d(np.where(filtered_simplexes.T[x[0]] == 1), added_complex)) > 0]

        # Are there enough simplexes to make a new one?
        if tuples.shape[0] > dim:

            # We are looking for all combinations that consist of len(added_complex) + 1, but since we only care about
            # combinations where added_complex is included, we iterate over all combinations of len(added_complex)
            # tuples, and then add the added complex to each combination.
            for group in combinations(filtered_tuples, dim):
                group = np.array(group)
                indices = np.append(group[:, 0], added_complex_index)  # indices of simplexes in the combination
                uniques = np.unique(indices)  # Remove duplicate entries

                simplexes = np.concatenate((group[:, 1:], [added_complex]))

                # If the combination is still of proper length, and doesn't already exist add it as newly created
                # simplex
                if uniques.shape[0] == dim + 1 and is_valid_simplex(simplexes):
                    new_complexes.append(indices)
        return new_complexes


def is_valid_simplex(index_tuples):
    """Check whether a set of simplexes (in the form of lists of indices) are a valid simplex

    :param index_tuples: A list of lists of indices that make up the simplexes.
    :return: True if the combination of tuples is a valid simplex, False otherwise.
    """
    flattened = index_tuples.flatten()
    occurrences = set([np.sum(flattened == x) for x in np.unique(flattened)])

    # Every simplex should appear exactly twice
    return len(occurrences) == 1 and occurrences.pop() == 2


def simplex_exists(simplex, filtered_simplexes):
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


def all_simplexes_found(zero_simplex_count, filtered_simplexes, max_dimensions):
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


def to_index_tuples(filtered_simplexes, dimension_filter=None):
    """Get all simplexes from a filtered simplex matrix as tuples of indices of the simplexes that created them.

    :param filtered_simplexes: A filtered simplex matrix.
    :param dimension_filter: Optional, default None - An optional filter to only select simplexes of a given
        dimensionality
    :return: A list of (tuples with shape [index of simplex in filtered_simplexes, ..indices of simplexes that make up
        this simplex])
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


class HbdscanRips:
    """A simple wrapper function that applies HBDSCAN over a distance matrix before passing it through the rips()
    function.

    :param max_dimensions: Optional, default 2 - The upper bound of dimensions for simplexes created.
    :param max_radius: Optional, default np.inf (infinity) - The maximum
    :param k_core: The neighbourhood size (K) to be used to calculate the core distances.
    """

    def __init__(self, max_dimensions=2, max_radius=np.inf, k_core=5):
        """Initialize the class, and create an instance of HbdscanRips.

        :param max_dimensions: Optional, default 2 - The upper bound of dimensions for simplexes created.
        :param max_radius: Optional, default np.inf (infinity) - The maximum
        :param k_core: The neighbourhood size (K) to be used to calculate the core distances.
        """
        self.rips = Rips(max_dimensions, max_radius)
        self.k_core = k_core

    def fit(self, distance_matrix):
        """Fit the class to the distance matrix.

        :param distance_matrix: A distance matrix of a set of points.
        :return: This class instance.
        """
        self.rips.fit(distance_matrix)
        return self

    def transform(self, distance_matrix):
        """Applies the Rips algorithm over the core distances of a given distance matrix to produce a filtered
        simplicial complex. This algorithm terminates either when all simplexes are found, or when the maximum radius is
        reached.

        :param distance_matrix: A distance matrix of a set of points
        :return: A filtered complex matrix, and a list with the corresponding birth radii.
        """
        altered_distances = np.zeros(distance_matrix.shape)
        for i in range(distance_matrix.shape[0]):
            for j in range(i + 1, distance_matrix.shape[0]):
                i_core = self.__core_distance(distance_matrix, i, self.k_core)
                j_core = self.__core_distance(distance_matrix, j, self.k_core)
                altered_distances[i][j] = altered_distances[j][i] = max(i_core, j_core, distance_matrix[i][j])
        return self.rips.transform(altered_distances)

    def __core_distance(self, distance_matrix, origin_index, k):
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
    used = set()
    reduced = reduce_columns(filtered_complexes)

    for i, col in enumerate(reduced.T):
        if np.count_nonzero(col) > 0:
            death = labels[i]
            j = np.max(np.where(col == 1))
            if len(np.where(col == 1)) > 2:
                print(f"non-zero {np.where(col == 1).shape} - {i} - {labels[i]} - {j} - {labels[j]}")
            birth = labels[j]
            used.add(i)
            used.add(j)
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

    :param resolution: The size of the resulting matrix (Integer); result will be matrix of shape (resolution,
        resolution).
    :param kernel_spread: Optional, default 2 - Defines the kernel-spread used to calculate density.
    """

    def __init__(self, resolution, kernel_spread=2):
        self.resolution = resolution
        self.kernel_spread = kernel_spread
        self.max_weight = 1
        self.points = None
        self.x_min = self.x_max = self.y_min = self.y_max = None

    def fit(self, filtered_complexes, labels, x_min=None, x_max=None, y_min=None, y_max=None):
        """Fits the PersistenceImage to the filtered simplicial complex. Pre-processes the filtered complex matrix and
        labels so a persistence image can be generated.

        :param filtered_complexes: A filtered complex matrix.
        :param labels: The birth-labels corresponding to the filtered complex matrix.
        :param x_min: Optional, default None - Lower bound for the x-axis, if not defined: 0.
        :param x_max: Optional, default None - Upper bound for the x-axis, if not defined: Maximum x value + 0.5.
        :param y_min: Optional, default None - Lower bound for the y-axis, if not defined: 0.
        :param y_max: Optional, default None - Upper bound for the y-axis, if not defined: Maximum y value + 0.5.
        :return: This class instance.
        """
        tuples = filtered_complexes_to_tuples(filtered_complexes, labels)
        # Replace death = infinity with 2x the greatest death value
        masked_tuples = np.ma.array(tuples, mask=~np.isfinite(tuples)).flatten()
        max_idx = np.ma.argmax(masked_tuples, fill_value=-np.inf)  # <- The index of the simplex with highest death value that isn't infinity
        self.points = transform_to_birth_persistence(tuples, infinity_replacement=2 * tuples.flatten()[max_idx])

        if x_min is None:
            self.x_min = 0
        else:
            self.x_min = x_min

        if x_max is None:
            self.x_max = np.max(self.points.T[0]) + .5
        else:
            self.x_max = x_max

        if y_min is None:
            self.y_min = 0
        else:
            self.y_min = y_min

        if y_max is None:
            self.y_max = np.max(self.points.T[1]) + .5
        else:
            self.y_max = y_max

        return self

    def transform(self, filtered_complexes, labels):
        """Generates a persistence image with a given resolution for the dataset.

        :return: A matrix of shape (resolution, resolution) with describing the density at each pixel.
        """
        pixels = self.__to_pixels(self.resolution, self.x_min, self.x_max, self.y_min, self.y_max, self.kernel_spread)
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

    :param steps: The amount of steps in the discretization step.
    """

    def __init__(self, steps):
        self.steps = None
        self.x_min = None
        self.x_max = None
        self.points = None
        self.steps = steps

    def fit(self, filtered_complexes, labels, x_min=None, x_max=None):
        """Fits the PersistenceLandscape class. Pre-processes the filtered complex matrix and labels so a
        persistence landscape can be generated.

        :param filtered_complexes: A filtered complex matrix.
        :param labels: The birth-labels corresponding to the filtered complex matrix.
        :param x_min: Optional, default None - The lower bound of the x-axis, if not specified the lowest x of the
            dataset is used.
        :param x_max: Optional, default None - The upper bound of the x-axis, if not specified the highest x of the
            dataset is used.
        :return: This class instance.
        """
        tuples = filtered_complexes_to_tuples(filtered_complexes, labels)
        # Replace death = infinity with 2x the greatest death value
        masked_tuples = np.ma.array(tuples, mask=~np.isfinite(tuples)).flatten()
        max_idx = np.ma.argmax(masked_tuples, fill_value=-np.inf)
        self.points = transform_to_mid_half_life(tuples, infinity_replacement=2 * tuples.flatten()[max_idx])

        if x_min is None:
            self.x_min = min(self.points, key=lambda x: x[0])[0]
        else:
            self.x_min = x_min
        if x_max is None:
            self.x_max = max(self.points, key=lambda x: x[0])[0]
        else:
            self.x_max = x_max
        return self

    def transform(self, filtered_complexes, labels):
        """Generate a persistence landscape at a given resolution for the dataset.

        :param filtered_complexes: A filtered complex matrix.
        :param labels: The birth-labels corresponding to the filtered complex matrix.
        :return: A 2 dimensional numpy array with the persistence landscape functions.
        """
        result = np.zeros((len(self.points), self.steps))

        step_size = (self.x_max - self.x_min) / self.steps
        for i in range(self.steps):
            x = step_size * i + self.x_min
            for y, point in enumerate(self.points):
                result[y, i] = self.__get_triangle_height(x, point)

        result = np.flip(np.sort(result, axis=0), axis=0)

        return (self.x_min, self.x_max), result

    def __get_triangle_height(self, x, peak):
        """Get the height of a triangle at a given point on the x axis.

        :param x: A point on the x-axis.
        :param peak: A tuple of (x, y) coordinates.
        :return: The height of the triangle with a given peak at x. Triangles have slope 1.
        """
        # Simply subtract x-axis distance from the height, since all triangles have slope 1
        distance = math.fabs(peak[0] - x)
        return max(0, peak[1] - distance)

