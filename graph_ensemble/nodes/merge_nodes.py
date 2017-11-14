import numpy as np
from abc import ABCMeta, abstractmethod
from ..core import Node


class Merge(Node):
    """
    Abstract class for nodes that take multiple nodes as input and merge
    their outputs into one output.

    Parameters
    ----------
    inputs : list or None
        A list of two or more nodes to use as inputs.
        The default is None, in which case the inputs can be set later
        using set_input.

    Attributes
    ----------
    _input : list
        The list of input nodes.
    """
    __metaclass__ = ABCMeta

    def __init__(self, inputs=None):
        super(Merge, self).__init__()
        self._input = None
        self.set_input(inputs)

    def add_input(self, input_node):
        """
        Append an input node to the inputs of the merge node.

        Parameters
        ----------
        input_node : Node
            The node to add as input

        Raises
        ------
        TypeError
            If the given node is not of a valid Node type.
        """
        Node.validate_type(input_node)
        self._input.append(input_node)

    def set_input(self, inputs):
        """
        Set the inputs of the node to a given list of input nodes or None.

        Parameters
        ----------
        inputs : list or None
            A list of two or more nodes to use as inputs.
            None will reset the node's input to None.

        Raises
        ------
        TypeError
            If inputs is not of type list or None or if any of the nodes
            is not of valid Node type.
        ValueError
            If less than 2 nodes are given.
        """
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
        """
        Abstract. Function used to merge the inputs, depending on the definition of the merge function.

        Parameters
        ----------
        arrays : array-like
            The input arrays to perform the merge on.

        Returns
        -------
        array-like
            The merged input.
        """
        pass

    def get_output(self):
        """
        Gets the input from the input nodes and merges them using the merge function.

        Returns
        -------
        array-like
            The merged input.
        """
        if self._output is None:
            outputs = [node.get_output() for node in self._input]
            self._output = self._merge_function(outputs)
        return self._output

    def fit(self, y_true):
        """
        By default, merge nodes are not trainable, so this method does not affect the
        merge node itself, and just trains its input nodes. This behavior can be overriden.
        Of course, merge nodes must fit their input nodes.

        Parameters
        ----------
        y_true : object (typically array-like)
            The target values used for training
        """
        for node in self._input:
            node.fit(y_true)


class Concatenate(Merge):
    """
    Concatenate the inputs on a given axis.

    Parameters
    ----------
    inputs : list or None
        A list of two or more nodes to use as inputs.
        The default is None, in which case the inputs can be set later
        using set_input.
    axis : int
        The axis to perform the concatenation on. The default is 0.

    Attributes
    ----------
    _axis : int
        The axis to perform the concatenation on.
    """
    def __init__(self, inputs=None, axis=0):
        super(Concatenate, self).__init__(inputs=inputs)
        self._axis = axis

    def _merge_function(self, arrays):
        return np.concatenate(arrays, axis=self._axis)


class Sum(Merge):
    """
    Sum the inputs on a given axis.

    Parameters
    ----------
    inputs : list or None
        A list of two or more nodes to use as inputs.
        The default is None, in which case the inputs can be set later
        using set_input.
    axis : int
        The axis to perform the sum on. The default is 0.

    Attributes
    ----------
    _axis : int
        The axis to perform the sum on.
    """
    def __init__(self, inputs=None, axis=None):
        super(Sum, self).__init__(inputs=inputs)
        self._axis = axis

    def _merge_function(self, arrays):
        return np.sum(arrays, axis=self._axis)


class Dot(Merge):
    """
    Perform dot product on the inputs.

    Parameters
    ----------
    inputs : list or None
        A list of two or more nodes to use as inputs.
        The default is None, in which case the inputs can be set later
        using set_input.
    """
    def __init__(self, inputs=None):
        super(Dot, self).__init__(inputs=inputs)

    def _merge_function(self, arrays):
        return reduce(np.dot, arrays)


class TensorDot(Merge):
    """
    Perform dot product on the inputs at a given axis.

    Parameters
    ----------
    inputs : list or None
        A list of two or more nodes to use as inputs.
        The default is None, in which case the inputs can be set later
        using set_input.
    axis : int
        The axis to perform the dot product on. The default is 2.

    Attributes
    ----------
    _axis : int
        The axis to perform the dot on.
    """
    def __init__(self, inputs=None, axis=2):
        super(TensorDot, self).__init__(inputs=inputs)
        self._axis = axis

    def _tensordot(self, a, b):
        return np.tensordot(a, b, axes=self._axis)

    def _merge_function(self, arrays):
        return reduce(self._tensordot, arrays)


class Mean(Merge):
    """
    Returns the mean of the inputs at a given axis.

    Parameters
    ----------
    inputs : list or None
        A list of two or more nodes to use as inputs.
        The default is None, in which case the inputs can be set later
        using set_input.
    axis : int
        The axis or axes along which the means are computed.
        The default is None, which would result in the mean of the entire input.

    Attributes
    ----------
    _axis : int
        The axis or axes along which the means are computed.
    """
    def __init__(self, inputs=None, axis=None):
        super(Mean, self).__init__(inputs=inputs)
        self._axis = axis

    def _merge_function(self, arrays):
        return np.mean(arrays, axis=self._axis)


class WeightedAverage(Merge):
    """
    Returns the weighted average of the inputs at a given axis.

    Parameters
    ----------
    inputs : list or None
        A list of two or more nodes to use as inputs.
        The default is None, in which case the inputs can be set later
        using set_input.
    axis : int
        The axis or axes along which the averages are computed.
        The default is None, which would result in the average of the entire input.
    weights : array-like
        The weights to assign to each input/value.
        The default is None, which means all inputs/values are of equal weight.

    Attributes
    ----------
    _axis : int
        The axis or axes along which the averages are computed.
    _weights : array-like
        The weights to assign to each input/value.
    """
    def __init__(self, inputs=None, axis=None, weights=None):
        super(WeightedAverage, self).__init__(inputs=inputs)
        self._axis = axis
        self._weights = weights

    def _merge_function(self, arrays):
        return np.average(arrays, axis=self._axis, weights=self._weights)


class Median(Merge):
    """
    Returns the median of the inputs at a given axis.

    Parameters
    ----------
    inputs : list or None
        A list of two or more nodes to use as inputs.
        The default is None, in which case the inputs can be set later
        using set_input.
    axis : int
        The axis or axes along which the medians are computed.
        The default is None, which would result in the median of the entire input.

    Attributes
    ----------
    _axis : int
        The axis or axes along which the medians are computed.
    """
    def __init__(self, inputs=None, axis=None):
        super(Median, self).__init__(inputs=inputs)
        self._axis = axis

    def _merge_function(self, arrays):
        return np.median(arrays, axis=self._axis)

