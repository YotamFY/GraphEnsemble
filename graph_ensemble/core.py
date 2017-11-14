import numpy as np
from abc import ABCMeta, abstractmethod


class Node(object):
    """
    Abstract base class for all node types.

    Attributes
    ----------
    _input : Node
        The node the current node gets input from.
        None by default, needs to be set using set_input(). See set_input below.
    _output : object
        The output of the current node.
    _fitted : bool
        Indicates whether the current node has already been fitted.
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        self._input = None
        self._output = None
        self._fitted = False

    @staticmethod
    def validate_type(node):
        """
        Checks whether a given object is a Node object.

        Parameters
        ----------
        node : Node
            The object to validate

        Raises
        ------
        TypeError
            When the given object is not an instance of Node
        """
        if not isinstance(node, Node):
            raise TypeError('Unrecognized node type: %s' % type(node))

    def set_input(self, inp):
        """
        Sets the current node's input to a given object.

        Parameters
        ----------
        inp : Node
            The node to be set as the current node's input

        Raises
        ------
        TypeError
            When the given object is not an instance of Node
        """
        Node.validate_type(inp)
        self._input = inp

    def clear(self):
        """Resets the node's status (output and fitted flag)"""
        self._output = None
        self._fitted = False

    @property
    def input(self):
        """
        Retrives the node's input

        Returns
        -------
        Node
            The input node of the current node
        """
        return self._input

    @abstractmethod
    def get_output(self):
        """Abstract. Should return the node's current output."""
        pass

    def fit(self, y):
        """
        Fits the node on data provided by its input node and user-provided labels.
        If the input hasn't been fitted yet, it will be fitted before fetching data from it.

        NOTE: The default behavior is, if the node hasn't been fit yet, to fit the input,
        then fetch data from it and fit the current node on it by calling _fit(X, y), and set the _fitted flag to True.
        Some nodes might override this behaviour.
        Typically, the behavior outside of _fit is necessary (training the input nodes and updating the state),
        so if you want to re-define training, usually overriding _fit is enough. Although in some cases (e.g. merge nodes)
        this method needs to be overriden as well.

        Parameters
        ----------
        y : object
            The labels to fit the nodes on.
        """
        if not self._fitted:
            self._input.fit(y)
            train_data = self._input.get_output()
            self._fit(train_data, y)
            self._fitted = True

    def _fit(self, X, y):
        """
        The actuall training procedure for the node.
        The default behavior is to do nothing. If your node is to be trainable,
        you should override this method.

        Parameters
        ----------
        X : object
            Input data
        y : object
            Labels
        """
        pass


class Input(Node):
    """
    A Node to be used as a placeholder for the graph's input.
    """
    def __init__(self):
        super(Input, self).__init__()

    def store(self, inp):
        """
        stores the input in the node.

        Parameters
        ----------
        inp : object
            The object to be used as input
        """
        self._output = inp

    def get_output(self):
        """
        Returns the input stored in the node.

        Returns
        -------
        object
            The input stored in the node
        """
        return self._output

    def fit(self, *args, **kwargs):
        """
        Does nothing. This node is untrainable.
        """
        pass


class Graph(object):
    """
    A Graph object encompases path(s) of nodes (Node objects)
    starting at the input node(s) and ending at an output node.
    At each session of the graph (see below), the data fed to the graph
    is used to train the nodes of the nodes or make predictions in the following manner:
    Each node gets the output of its input node (see Node above) as input.
    A node's output is node-dependent. For example, it might be the predictions of some model
    on its input data.
    The firstmost node(s) the data is fed to is/are the input node(s) (see Input above).
    The output of the graph is the output of its output node.

    Parameters
    ----------
    input_nodes : Input or list
        The input of the graph. Can be either a node of type Input if the graph
        has only one input, or a list of Input nodes if the graph has multiple inputs.
    output_node : Node
        The output node of the graph.

    Attributes
    ----------
    _input_nodes : list
        A list containing the input nodes. (Even if there's only a single input!)
    _output_node : Node
        The output node of the graph.

    Raises
    ------
    TypeError
        If (one or more of) the input(s) is not of type Input or the output node is
        not of type Node.

    Notes
    -----
    Multi-output graphs are not yet supported, but will be in the future.
    Cross-validation support is to be added soon.
    """
    def __init__(self, input_nodes, output_node):
        self._input_nodes = Graph._get_node_list(input_nodes)
        Node.validate_type(output_node)
        self._output_node = output_node

    @staticmethod
    def _get_node_list(nodes):
        if type(nodes) is not list:
            nodes = [nodes]
        node_list = []
        for node in nodes:
            if isinstance(node, Input):
                node_list.append(node)
            else:
                types = ','.join(str(type(obj)) for obj in nodes)
                raise TypeError('All inputs must be instances of Input. Got %s' % types)
        return node_list

    @property
    def input_nodes(self):
        """
        The input nodes of the graph.

        Returns
        -------
        list
            A list of the graph's input nodes
        """
        return self._input_nodes[:]

    @property
    def output_node(self):
        """
        The output node of the graph.

        Returns
        -------
        Node
            The output node of the graph.
        """
        return self._output_node

    def _validate_input(self, inp, nodes):
        if type(inp) is tuple:
            # For multiple inputs, the length of the tuple must
            # match the number of input nodes.
            if len(inp) != len(nodes):
                raise ValueError('Number of inputs (%d) does not match the number of input nodes (%d)' % (len(inp), len(nodes)))
        else:
            # For a single input, all of the graph's input nodes are fed
            # the same data (this works for a single input node in particular)
            # by creating a tuple of length <n_input_nodes> where each element is
            # the input object.
            inp = (inp,) * len(nodes)
        return inp

    def _set_inputs(self, X):
        # Store each input object in its respective input node.
        for inp_node, X_arr in zip(self._input_nodes, X):
            inp_node.store(X_arr)

    def fit(self, X, y):
        """
        Fit the graph's nodes on given training data.

        Parameters
        ----------
        X : object (typically array-like) or tuple
            Training features.
            Tuple of input objects for multiple inputs. In this case, the tuple's length must match
            the number of input nodes.
            Otherwise, the same input object is fed to all of the graph's input nodes, and for the graph's
            single input node in particular if it has only one input.
            Input objects are typically array-like.
        y : object (typically array-like)
            Target values.

        Returns
        -------
        Graph
            The graph object

        Raises
        ------
        ValueError
            If the number of inputs does not match the number of input nodes.
        """
        # Validate input data
        X = self._validate_input(X, self._input_nodes)
        # Prepare the input nodes for the training session
        self._set_inputs(X)
        # Start the training. Starting at the output node, each nodes "requests" its input
        # from its input node(s) until the graph's input node(s) are reached.
        self._output_node.fit(y)
        # Reset the nodes' states and make them ready for the next session.
        self._clear_nodes()
        return self

    def _clear_path(self, node):
        # A node at the end of a path has no input node.
        if node is not None:
            # Reset the current node
            node.clear()
            # Clear the input node(s) of the current node.
            # Some nodes might have multiple inputs. Here, it must be supported.
            if type(node.input) is list:
                for inp in node.input:
                    self._clear_path(inp)
            else:
                self._clear_path(node.input)

    def _clear_nodes(self):
        # Walk through each of the paths in the graph and
        # clear the states of the nodes.
        self._clear_path(self._output_node)

    def predict(self, X):
        """
        Make a prediction using the graph on given data.

        Parameters
        ----------
        X : object (typically array-like) or tuple
            Features to predict on.
            Tuple of input objects for multiple inputs. In this case, the tuple's length must match
            the number of input nodes.
            Otherwise, the same input object is fed to all of the graph's input nodes, and for the graph's
            single input node in particular if it has only one input.
            Input objects are typically array-like.

        Returns
        -------
        object (typically array-like)
            The graph's predictions on X.

        Raises
        ------
        ValueError
            If the number of inputs does not match the number of input nodes.
        """
        # Validate the input data and make it feedable to the graph
        X = self._validate_input(X, self._input_nodes)
        # Prepare the input nodes for the predict session
        self._set_inputs(X)
        # Get the output of the output node as final predictions
        preds = self._output_node.get_output()
        # Reset the nodes to make them ready for the next session
        self._clear_nodes()
        return preds

