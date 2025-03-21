"""
TWED distance computation based on Numba JIT
"""


import numpy as np
import pandas as pd
from .utils import (_twed, _backtracking)


def twed(a, b, ts_a=None, ts_b=None, p=2, nu=0.001, lam=1.0, path_out=False):
    """
    Compute the Time Warp Edit Distance (TWED) between two time series.

    This function provides a unified interface for calculating TWED. The time series may be provided
    either as pandas DataFrames or as numpy arrays. For pandas DataFrames, the index is used as the time
    stamps. For numpy arrays, if time stamp arrays (`ts_a`, `ts_b`) are not provided, default sequential
    time stamps are generated.

    :param a: First time series, provided as a pandas DataFrame or a numpy array.
    :param b: Second time series, provided as a pandas DataFrame or a numpy array.
    :param ts_a: Optional time stamps for the first time series if `a` is a numpy array. Defaults to None.
    :param ts_b: Optional time stamps for the second time series if `b` is a numpy array. Defaults to None.
    :param p: The l_p distance parameter (default is p=2).
    :param nu: Non-negative float representing the penalty for temporal differences (default is 0.001).
    :param lam: Non-negative float representing the penalty for deletion operations (default is 1.0).
    :param path_out: If True, returns a tuple (TWED, edit_path); otherwise, only the TWED value is returned.
    :return: The computed TWED value, or a tuple (TWED, edit_path) if `path_out` is True.
    :raises TypeError: If the input types for `a` and `b` are not both pandas DataFrames or numpy arrays.
    """

    # Check if both inputs are DataFrames
    if isinstance(a, pd.DataFrame) and isinstance(b, pd.DataFrame):
        return _twed_dataframe(a, b, p, nu, lam, path_out)

    # Check if both inputs are not DataFrames (assume numpy arrays or similar)
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return _twed_ndarray(a, b, ts_a, ts_b, p, nu, lam, path_out)

    # If one is a DataFrame and the other is not, raise an error.
    raise TypeError("Both inputs must be either pandas DataFrames or numpy arrays.")


def _twed_dataframe(df_a, df_b, p, nu, lam, path_out):
    """
    Compute the TWED between two time series provided as pandas DataFrames.

    This helper function extracts numerical data and time stamps from the DataFrames, and computes the TWED
    using an underlying function. Optionally, it returns the optimal edit path determined via backtracking.

    :param df_a: First time series as a pandas DataFrame.
    :param df_b: Second time series as a pandas DataFrame.
    :param p: The l_p distance parameter (p).
    :param nu: Non-negative float representing the temporal penalty.
    :param lam: Non-negative float representing the deletion penalty.
    :param path_out: Boolean flag indicating whether to return the edit path along with the TWED.
    :return: The TWED value, or a tuple (TWED, edit_path) if `path_out` is True.
    """

    # Extract data and time indices from DataFrames
    # and process them with the backend TWED function
    out = _twed(df_a.values.astype('float32'),
                df_b.values.astype('float32'),
                np.array(df_a.index, dtype='float32'),
                np.array(df_b.index, dtype='float32'),
                p, nu, lam)

    if path_out:
        return out[0], _backtracking(out[1])

    return out[0]


def _twed_ndarray(a, b, ts_a, ts_b, p, nu, lam, path_out):
    """
    Compute the TWED between two time series provided as numpy arrays.

    This helper function computes the TWED for time series given as numpy arrays. If time stamps (`ts_a`, `ts_b`)
    are not provided, default sequential indices are generated. Optionally, it returns the optimal edit path via
    backtracking.

    :param a: First time series as a numpy array.
    :param b: Second time series as a numpy array.
    :param ts_a: Optional numpy array of time stamps for the first time series. If None, a default range is used.
    :param ts_b: Optional numpy array of time stamps for the second time series. If None, a default range is used.
    :param p: The l_p distance parameter (p).
    :param nu: Non-negative float representing the temporal penalty.
    :param lam: Non-negative float representing the deletion penalty.
    :param path_out: Boolean flag indicating whether to return the optimal edit path along with the TWED.
    :return: The TWED value, or a tuple (TWED, edit_path) if `path_out` is True.
    """

    # Inputs must be numpy arrays
    if ts_a is None:
        ts_a = np.arange(a.shape[0], dtype='float32')
    if ts_b is None:
        ts_b = np.arange(b.shape[0], dtype='float32')

    out = _twed(a, b, ts_a, ts_b, p, nu, lam)

    if path_out:
        return out[0], _backtracking(out[1])

    return out[0]
