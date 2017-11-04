import numpy as np
from ..core import Node


class Reshape(Node):
    def __init__(self, new_shape):
        super(Reshape, self).__init__()
        self._new_shape = new_shape

    def get_output(self):
        if self._output is None:
            self._output = self._input.get_output().reshape(self._new_shape)
        return self._output

