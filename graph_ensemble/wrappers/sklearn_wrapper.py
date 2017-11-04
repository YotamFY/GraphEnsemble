from ..core import Node


class SKLearnNode(Node):
    def __init__(self, model, use_probas=False):
        super(SKLearnNode, self).__init__()
        self._model = model
        self._use_probas = use_probas

    def _fit(self, X, y):
        self._model = self._model.fit(X, y)

    def get_output(self):
        if self._output is None:
            X = self._input.get_output()
            if self._use_probas:
                self._output = self._model.predict_proba(X)
            else:
                self._output = self._model.predict(X)
        return self._output

