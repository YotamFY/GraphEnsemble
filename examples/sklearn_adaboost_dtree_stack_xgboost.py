from graph_ensemble import Graph
from graph_ensemble.nodes import Concatenate
from graph_ensemble.wrappers import SKLearnNode, Input
from xgboost import XGBRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import datasets


X, y = datasets.load_boston(return_X_y=True)
print y.shape


inp = Input()
ab = SKLearnNode(AdaBoostRegressor())
ab.set_input(inp)
dtree = SKLearnNode(DecisionTreeRegressor())
dtree.set_input(inp)
concat = Concatenate([ab, dtree])

graph = Graph(inp, concat)

graph.fit(X, y)
print graph.predict(X).shape

