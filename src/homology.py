import numpy as np


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
