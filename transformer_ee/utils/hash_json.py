"""
Converts a dictionary to a string.
"""
# https://www.doc.ic.ac.uk/~nuric/coding/how-to-hash-a-dictionary-in-python.html

import hashlib
import json

def hash_dict(d: dict) -> str:
    r"""
    Hash a dictionary.

    Parameters
    ----------
    d : dict
        The dictionary to be hashed.

    Returns
    -------
    str
        The hash of the dictionary.
    """
    return hashlib.md5(json.dumps(d, sort_keys=True).encode("utf-8")).hexdigest()
