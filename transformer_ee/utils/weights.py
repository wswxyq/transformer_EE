"""
Calculate the weights for the loss function.
"""

import numpy as np


class NullWeights:
    """
    A null weighter.
    """

    def getweight(self, _):
        """
        Return 1 for all weights.
        """
        return 1.0


class FlatSpectraWeights:
    """
    Initialize the weights with an array, then return the weights (as array) for a given value x.
    The weighter will calculate the weight based on the following function using the histogram count:
        w = 1 / (count + alpha * max (count))
    In addition, the weighter will normalize the weights to have a mean of 1 and clip the weights to a given range.
    """

    def __init__(  # pylint: disable=W0622
        self, array, bins=50, range=None, maxweight=np.inf, minweight=-np.inf, alpha=0
    ) -> None:
        self.hist = np.histogram(array, bins=bins, range=range)
        _alpha = max(self.hist[0]) * alpha
        if _alpha < 1:
            _alpha = 1
        self.weights = 1 / (
            self.hist[0] + _alpha
        )  # Add _alpha > 0 to avoid division by zero
        self.weights = self.weights / np.mean(self.weights)
        self.bins = self.hist[1]
        self.weights[self.weights > maxweight] = maxweight
        self.weights[self.weights < minweight] = minweight
        self.bins[0] = -np.inf
        self.bins[-1] = np.inf

    def getweight(self, x):
        return self.weights[np.digitize(x, self.bins) - 1]


def create_weighter(config: dict, array):
    """
    Create a weighter from a config.
    Args:
        "config":
        {
            "weight":
            {
                name: the name of weighter.
                kwargs: the arguments of weighter.
                    For FlatSpectraWeights, kwargs should be:
                    {
                        "bins": the number of bins for the histogram.
                        "range": the range of the histogram. eg. (0, 5)
                        ...
                    }
            ...
            }
        }
    """
    if "weight" not in config:
        return NullWeights()

    _kwgs = config["weight"].get("kwargs", {})
    if config["weight"]["name"] == "FlatSpectraWeights":
        weighter = FlatSpectraWeights(array, **_kwgs)
    else:
        raise ValueError(f"Unsupported weighter: {config['weight']['name']}")
    return weighter
