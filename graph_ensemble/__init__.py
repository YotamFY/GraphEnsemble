"""
GraphEnsemble is a module for creating ensembles of learners and meta-learners
in the form of an arbitrary graph.
The concept is a generalization of stacking ensembles into multi-level and multi-path graphs.
Each node learns the predcitions of its input node(s) and provides its predictions to the succeeding node(s) as training data.
Graphs can be of any depth and width. In addition, this package provides utility nodes such as several merging nodes and shape manipulation.
"""

from core import *

