from ..core import Node


class SKLearnNode(Node):
    """
    Encompases a scikit-learn compatible model.
    The model must have fit and predict methods.
    Training a node of this type will train its
    internal model.
    Its output is the predictions of the model
    on the given features.

    Parameters
    ----------
    model : scikit-learn model
        A scikit-learn compatible model object.
        It must have fit and predict methods.
    use_probas : bool
        For classification, output the probability of each class for each
        sample if True, otherwise output the class values.
        The default is False.
        In addition, if True, the model must have a predict_proba method.

    Attributes
    ----------
    _model : scikit-learn model
        A scikit-learn compatible model object.
        It must have fit and predict methods.
    _use_probas : bool
        For classification, output the probability of each class for each
        sample if True, otherwise output the class values.
        In addition, if True, the model must have a predict_proba method.
    """
    def __init__(self, model, use_probas=False):
        super(SKLearnNode, self).__init__()
        self._model = model
        self._use_probas = use_probas

    def _fit(self, X, y):
        # Fit the model on features X with targets y
        self._model = self._model.fit(X, y)

    def get_output(self):
        """
        Returns
        -------
        array-like
            The predictions of the model.
            For classification, the output will be a 2D array of
            Probabilities for each class if the use_probas flag
            is set to True, otherwise class values.
        """
        if self._output is None:
            X = self._input.get_output()
            if self._use_probas:
                self._output = self._model.predict_proba(X)
            else:
                self._output = self._model.predict(X)
        return self._output

