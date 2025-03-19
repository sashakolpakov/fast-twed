import numpy as np
from numba import jit

# Padding functions for time series arrays


def mypad1s(ar):
    res = np.zeros(ar.shape[0]+1)
    res[1:res.shape[0]] = ar
    return res


def mypad2s(ar):
    res = np.zeros((ar.shape[0]+1, ar.shape[1]))
    res[1:res.shape[0],:] = ar
    return res


# jitified padding functions
mypad1s_jit = jit(mypad1s)
mypad2s_jit = jit(mypad2s)


@jit(nopython=True, fastmath=True)
def _dist_lp(a, b, p=2):
    """
    Compute the l_p distance between two arrays.

    :param a: First array.
    :param b: Second array.
    :param p: Order of the norm (default is 2).
    :return: The l_p distance between a and b.
    """
    return np.linalg.norm(a-b, p)


@jit(nopython=True, fastmath=True)
def _twed(a, b, ts_a, ts_b, nu, lam):
    """
    Compute the Time Warp Edit Distance (TWED) between two time series using dynamic programming.

    This function calculates the TWED between two numpy arrays representing time series data,
    taking into account both the magnitude differences and the temporal differences between the series.
    The calculation is accelerated using Numba's JIT compilation.

    The TWED is computed by constructing a dynamic programming matrix where each cell represents
    the minimal cost of aligning two subsequences. The cost function includes:
      - The l_p distance between consecutive points (computed via _dist_lp),
      - A penalty proportional to the absolute difference of time stamps (weighted by nu),
      - A fixed penalty (lam) for deletion operations.

    :param a: Numpy array for the first time series (expected shape: (m, ...)).
    :param b: Numpy array for the second time series (expected shape: (n, ...)).
    :param ts_a: Numpy array of time stamps corresponding to `a` (length m).
    :param ts_b: Numpy array of time stamps corresponding to `b` (length n).
    :param nu: Non-negative float; penalty for temporal differences between consecutive points.
    :param lam: Non-negative float; penalty for deletion operations.

    :return: A tuple containing:
             - The final TWED value (float), representing the distance between the two time series.
             - The dynamic programming matrix (numpy.ndarray) used to compute the TWED.

    :raises ValueError: If either `nu` or `lam` is negative.
    """

    if nu < 0:
        raise ValueError("Parameter nu must be non-negative.")

    if lam < 0:
        raise ValueError("Parameter lambda must be non-negative.")

    m = a.shape[0]
    a = mypad2s_jit(a)
    ts_a = mypad1s_jit(ts_a)

    n = b.shape[0]
    b = mypad2s_jit(b)
    ts_b = mypad1s_jit(ts_b)

    # Dynamical programming
    dyn = np.zeros((m, n))

    # Initialize the dynamic programming matrix
    # and set first row and column to infinity
    dyn[0, :] = np.inf
    dyn[:, 0] = np.inf
    dyn[0, 0] = 0.0

    # Compute minimal cost
    for i in range(1, m):
        for j in range(1, n):
            # Calculate and save cost of various operations
            c = np.ones((3, 1)) * np.inf
            # Deletion in A
            c[0] = (
                dyn[i - 1, j]
                + _dist_lp(a[i - 1], a[i])
                + nu * np.abs(ts_a[i] - ts_a[i - 1])
                + lam
            )
            # Deletion in B
            c[1] = (
                dyn[i, j - 1]
                + _dist_lp(b[j - 1], b[j])
                + nu * np.abs(ts_b[j] - ts_b[j - 1])
                + lam
            )
            # Keep data points in both time series
            ts_diff = (np.abs(ts_a[i] - ts_b[j]) +
                       np.abs(ts_a[i - 1] - ts_b[j - 1]))
            c[2] = (
                dyn[i - 1, j - 1]
                + _dist_lp(a[i], b[j])
                + _dist_lp(a[i - 1], b[j - 1])
                + nu * ts_diff
            )
            # Choose the operation with the minimal cost and update DP Matrix
            dyn[i, j] = np.min(c)

    return dyn[m - 1, n - 1], dyn


@jit(nopython=True, fastmath=True)
def _backtracking(dyn):
    """
    Compute the most cost-efficient edit path between two time series.

    This function performs a backtracking algorithm on the given dynamic programming matrix,
    typically generated during the computation of the Time Warp Edit Distance (TWED). It recovers
    the optimal sequence of edit operations (matches or deletions) that align the two time series.

    The returned path is a list of tuples, each containing a pair of indices corresponding to the
    aligned elements in the time series.

    :param dyn: A 2D numpy array representing the dynamic programming cost matrix.
    :type dyn: numpy.ndarray
    :return: A list of index pairs (tuples) representing the optimal edit path between the time series.
    :rtype: list[tuple]
    """

    dim = np.shape(dyn)
    i = dim[0] - 1
    j = dim[1] - 1

    # The indices of the paths are saved in the opposite direction
    best_path = []
    steps = 0

    while i != 0 or j != 0:
        best_path.append((i - 1, j - 1))

        c = np.ones((3, 1)) * np.inf

        # Keep data points in both time series
        c[0] = dyn[i - 1, j - 1]
        # Deletion in A
        c[1] = dyn[i - 1, j]
        # Deletion in B
        c[2] = dyn[i, j - 1]

        # Find the index for the lowest cost
        idx = np.argmin(c)

        if idx == 0:
            # Keep data points in both time series
            i = i - 1
            j = j - 1
        elif idx == 1:
            # Deletion in A
            i = i - 1
            j = j
        else:
            # Deletion in B
            i = i
            j = j - 1
        steps = steps + 1

    best_path.append((i - 1, j - 1))
    best_path.reverse()

    return best_path[1:]
