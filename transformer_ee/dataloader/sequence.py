"""Convert a string to a list of floats."""
import numpy as np


def string_to_float_list(string):
    """
    By default, convert all strings to list of floats.
    Otherwise, return a list of one zero.
    """
    if not string or not isinstance(string, str):
        return [0]
    return [float(s) for s in string.split(",")]


def sequence_statistics(column) -> list:
    """
    Calculate the mean and standard deviation of ALL ELEMENT in a sequence column.
    Causion: if std = 0, add a small value to avoid division by zero.
    """
    _ = []
    for sq in column:
        _.extend(sq) # flatten the list
    return [np.mean(_), np.std(_) + 1e-10] # 1e-10 is a small value to avoid division by zero
