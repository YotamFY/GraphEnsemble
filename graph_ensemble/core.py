import numpy as np
from abc import ABCMeta, abstractmethod


class Node(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        self._input = None
        self._output = None
        self._fitted = False

    @staticmethod
    def validate_type(node):
        if not isinstance(node, Node):
            raise TypeError('Unrecognized node type: %s' % type(node))

    def set_input(self, inp):
        Node.validate_type(inp)
        self._input = inp

    def clear(self):
        self._output = None
        self._fitted = False

    @property
    def input(self):
        return self._input

    @abstractmethod
    def get_output(self):
        pass

    def fit(self, y):
        if not self._fitted:
            self._input.fit(y)
            train_data = self._input.get_output()
            self._fit(train_data, y)
            self._fitted = True

    def _fit(self, X, y):
        pass


class Input(Node):
    def __init__(self):
        super(Input, self).__init__()

    def store(self, inp):
        self._output = inp

    def get_output(self):
        return self._output

    def fit(self, *args, **kwargs):
        pass


class Graph(object):
    def __init__(self, input_nodes, output_node):
        self._input_nodes = Graph._get_node_list(input_nodes)
        self._output_node = output_node

    @staticmethod
    def _get_node_list(nodes):
        if type(nodes) is not list:
            nodes = [nodes]
        node_list = []
        for node in nodes:
            Node.validate_type(node)
            node_list.append(node)
        return node_list

    @property
    def input_nodes(self):
        return self._input_nodes

    @property
    def output_node(self):
        return self._output_node

    def _validate_input(self, inp, nodes):
        if type(inp) is tuple:
            if len(inp) != len(nodes):
                raise ValueError('Number of inputs (%d) does not match the number of input nodes (%d)' % (len(inp), len(nodes)))
        else:
            inp = (inp,) * len(nodes)
        return inp

    def _set_inputs(self, X):
        for inp_node, X_arr in zip(self._input_nodes, X):
            inp_node.store(X_arr)

    def fit(self, X_train, y_train):
        X_train = self._validate_input(X_train, self._input_nodes)
        self._set_inputs(X_train)
        self._output_node.fit(y_train)
        self._clear_nodes()
        return self

    def _clear_path(self, node):
        if node is not None:
            node.clear()
            if type(node.input) is list:
                for inp in node.input:
                    self._clear_path(inp)
            else:
                self._clear_path(node.input)

    def _clear_nodes(self):
        self._clear_path(self._output_node)

    def predict(self, X):
        X = self._validate_input(X, self._input_nodes)
        self._set_inputs(X)
        preds = self._output_node.get_output()
        self._clear_nodes()
        if len(preds) == 1:
            preds = preds[0]
        return preds

