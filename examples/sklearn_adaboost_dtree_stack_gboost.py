from graph_ensemble import Graph
from graph_ensemble.nodes import Concatenate, Input, Reshape
from graph_ensemble.wrappers import SKLearnNode
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import datasets


X, y = datasets.load_boston(return_X_y=True)
print y.shape


inp = Input()
ab = SKLearnNode(AdaBoostRegressor())
ab.set_input(inp)
ab_reshape = Reshape((-1, 1))
ab_reshape.set_input(ab)
dtree = SKLearnNode(DecisionTreeRegressor())
dtree.set_input(inp)
dtree_reshape = Reshape((-1, 1))
dtree_reshape.set_input(dtree)
concat = Concatenate([ab_reshape, dtree_reshape], axis=-1)
gboost = SKLearnNode(GradientBoostingRegressor())
gboost.set_input(concat)

graph = Graph(inp, gboost)

graph.fit(X, y)
print graph.predict(X)

