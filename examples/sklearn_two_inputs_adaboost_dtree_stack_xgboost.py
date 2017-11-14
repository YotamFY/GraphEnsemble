from graph_ensemble import Graph
from graph_ensemble.nodes import Concatenate, Input, Reshape
from graph_ensemble.wrappers import SKLearnNode
from xgboost import XGBRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import datasets


X1, y = datasets.load_boston(return_X_y=True)
X2, _ = datasets.make_regression(n_samples=len(X1), n_features=5)
print y.shape


inp1 = Input()
inp2 = Input()
ab = SKLearnNode(AdaBoostRegressor())
ab.set_input(inp1)
ab_reshape = Reshape((-1, 1))
ab_reshape.set_input(ab)
dtree = SKLearnNode(DecisionTreeRegressor())
dtree.set_input(inp2)
dtree_reshape = Reshape((-1, 1))
dtree_reshape.set_input(dtree)
concat = Concatenate([ab_reshape, dtree_reshape], axis=-1)
xgbreg = SKLearnNode(XGBRegressor())
xgbreg.set_input(concat)

graph = Graph([inp1, inp2], xgbreg)

graph.fit((X1, X2), y)
print graph.predict((X1, X2))

