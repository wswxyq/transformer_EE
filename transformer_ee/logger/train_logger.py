"""
Abstract logger class that defines the interface for logging training information.
"""

from abc import ABC, abstractmethod


class BaseLogger(ABC):
    """
    Abstract logger class.
    """

    @abstractmethod
    def log_scalar(self, scalars: dict, step: int, epoch: int):  # pylint: disable=C0116
        pass

    @abstractmethod
    def close(self):  # pylint: disable=C0116
        pass
