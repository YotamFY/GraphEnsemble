import numpy as np
from abc import ABCMeta, abstractmethod
from ..core import Node


class Merge(Node):
    __metaclass__ = ABCMeta

    def __init__(self, inputs=None):
        super(Merge, self).__init__()
        self._input = None
        self.set_input(inputs)

    def add_input(self, input_node):
        Node.validate_type(input_node)
        self._input.append(input_node)

    def set_input(self, inputs):
        if inputs is None:
            self._input = None
        else:
            if type(inputs) is not list:
                raise TypeError('You need to provide None or a list of Node objects')
            elif len(inputs) < 2:
                raise ValueError('You need to provide at least 2 input nodes. Got %d' % len(inputs))
            for node in inputs:
                Node.validate_type(node)
            self._input = inputs[:]

    @abstractmethod
    def _merge_function(self, arrays):
        pass

    def get_output(self):
        if self._output is None:
            outputs = [node.get_output() for node in self._input]
            self._output = self._merge_function(outputs)
        return self._output

    def fit(self, y_true):
        for node in self._input:
            node.fit(y_true)


class Concatenate(Merge):
    def __init__(self, inputs=None, axis=0):
        super(Concatenate, self).__init__(inputs=inputs)
        self._axis = axis

    def _merge_function(self, arrays):
        return np.concatenate(arrays, axis=self._axis)


class Sum(Merge):
    def __init__(self, inputs=None, axis=None):
        super(Sum, self).__init__(inputs=inputs)
        self._axis = axis

    def _merge_function(self, arrays):
        return np.sum(arrays, axis=self._axis)


class Dot(Merge):
    def __init__(self, inputs=None):
        super(Dot, self).__init__(inputs=inputs)

    def _merge_function(self, arrays):
        return reduce(np.dot, arrays)


class TensorDot(Merge):
    def __init__(self, inputs=None, axis=2):
        super(TensorDot, self).__init__(inputs=inputs)
        self._axis = axis

    def _tensordot(self, a, b):
        return np.tensordot(a, b, axis=self._axis)

    def _merge_function(self, arrays):
        return reduce(self._tensordot, arrays)


class Mean(Merge):
    def __init__(self, inputs=None, axis=None):
        super(Mean, self).__init__(inputs=inputs)
        self._axis = axis

    def _merge_function(self, arrays):
        return np.mean(arrays, axis=self._axis)


class WeightedAverage(Merge):
    def __init__(inputs=None, axis=None, weights=None):
        super(WeightedAverage, self).__init__(inputs=inputs)
        self._axis = axis
        self._weights = weights

    def _merge_function(self, arrays):
        return np.average(arrays, axis=self._axis, weights=self._weights)


class Median(Merge):
    def __init__(inputs=None, axis=None):
        super(Median, self).__init__(inputs=inputs)
        self._axis = axis

    def _merge_function(self, arrays):
        return np.median(arrays, axis=self._axis)

