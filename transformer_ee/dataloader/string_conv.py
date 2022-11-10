"""Convert a string to a list of floats."""


def string_to_float_list(string):
    """
    By default, convert all strings to list of floats.
    Otherwise, return empty list.
    """
    if not string or not isinstance(string, str):
        return [0]
    return [float(s) for s in string.split(",")]
