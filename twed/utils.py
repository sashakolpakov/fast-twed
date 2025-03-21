"""
Utility functions for Fast TWED
"""


import numpy as np
from numba import jit, prange


# Padding functions for time series arrays


@jit(nopython=True)
def pad1d(ar):
    """
    :param ar: 1D Array
    :return: Array padded with zeros
    """
    res = np.zeros(ar.shape[0] + 1)
    res[1:res.shape[0]] = ar
    return res


@jit(nopython=True)
def pad2d(ar):
    """
    :param ar: 2D Array
    :return: Array padded with zeros
    """
    res = np.zeros((ar.shape[0] + 1, ar.shape[1]))
    res[1:res.shape[0], :] = ar
    return res


@jit(nopython=True, fastmath=True)
def _dist_lp(a, b, p=2):
    """
    Compute the l_p distance between two arrays.

    :param a: First array.
    :param b: Second array.
    :param p: Order of the norm (default is 2).
    :return: The l_p distance between a and b.
    """
    return np.sqrt(np.sum(np.abs(a - b) ** p))


@jit(nopython=True, fastmath=True)
def _twed(a, b, ts_a, ts_b, p, nu, lam):
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
    :param p: The l_p distance parameter (p).
    :param nu: Non-negative float; penalty for temporal differences between consecutive points.
    :param lam: Non-negative float; penalty for deletion operations.

    :return: A tuple containing:
             - The final TWED value (float), representing the distance between the two time series.
             - The dynamic programming matrix (numpy.ndarray) used to compute the TWED.

    :raises ValueError: If either `nu` or `lam` is negative.
    """

    if p < 1:
        raise ValueError("Parameter p must be at least 1.0.")

    if nu < 0:
        raise ValueError("Parameter nu must be non-negative.")

    if lam < 0:
        raise ValueError("Parameter lambda must be non-negative.")

    m = a.shape[0]
    a = pad2d(a)
    ts_a = pad1d(ts_a)

    n = b.shape[0]
    b = pad2d(b)
    ts_b = pad1d(ts_b)

    dyn = np.zeros((m, n))
    dyn[0, :] = np.inf
    dyn[:, 0] = np.inf
    dyn[0, 0] = 0.0

    for i in prange(1, m):
        for j in prange(1, n):
            # Compute cost for deletion in A
            cost_del_a = (
                    dyn[i - 1, j]
                    + _dist_lp(a[i - 1], a[i], p)
                    + nu * np.abs(ts_a[i] - ts_a[i - 1])
                    + lam
            )
            # Compute cost for deletion in B
            cost_del_b = (
                    dyn[i, j - 1]
                    + _dist_lp(b[j - 1], b[j], p)
                    + nu * np.abs(ts_b[j] - ts_b[j - 1])
                    + lam
            )
            # Compute cost for matching (keeping both points)
            ts_diff = np.abs(ts_a[i] - ts_b[j]) + np.abs(ts_a[i - 1] - ts_b[j - 1])
            cost_match = (
                    dyn[i - 1, j - 1]
                    + _dist_lp(a[i], b[j], p)
                    + _dist_lp(a[i - 1], b[j - 1], p)
                    + nu * ts_diff
            )
            # Update with the minimal cost using scalar comparisons
            if cost_del_a <= cost_del_b and cost_del_a <= cost_match:
                dyn[i, j] = cost_del_a
            elif cost_del_b <= cost_match:
                dyn[i, j] = cost_del_b
            else:
                dyn[i, j] = cost_match

    return dyn[m - 1, n - 1], dyn


@jit(nopython=True)
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

    best_path = []
    while i != 0 or j != 0:
        best_path.append((i - 1, j - 1))

        # Compute costs without creating an array
        cost_keep = dyn[i - 1, j - 1]
        cost_del_a = dyn[i - 1, j]
        cost_del_b = dyn[i, j - 1]

        # Determine the direction with the minimal cost
        if cost_keep <= cost_del_a and cost_keep <= cost_del_b:
            i -= 1
            j -= 1
        elif cost_del_a <= cost_del_b:
            i -= 1
        else:
            j -= 1

    best_path.append((i - 1, j - 1))
    best_path.reverse()
    return best_path[1:]
