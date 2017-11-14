import numpy as np
from ..core import Node


class Reshape(Node):
    """
    Transforms the input into a new shape without changing its data

    Parameters
    ----------
    new_shape : int or tuple of ints
        The new shape. It should be compatible with the original shape.
        If integer, the result will be 1D.
        If one of the dimensions is -1, its value will be inferred from the
        length and the remaining dimensions.
    """
    def __init__(self, new_shape):
        super(Reshape, self).__init__()
        self._new_shape = new_shape

    def get_output(self):
        """
        Returns
        -------
        array-like
            The output of the input node transformed into a new shape
        """
        if self._output is None:
            self._output = self._input.get_output().reshape(self._new_shape)
        return self._output

