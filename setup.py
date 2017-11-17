#!/usr/bin/env python

import os
import graph_ensemble


DEPENDENCIES = ['numpy', 'sklearn']

metadata = dict()

try:
    from setuptools import setup, find_packages
    packages  = find_packages()
    metadata['install_requires'] = DEPENDENCIES
except ImportError:
    from distutils.core import setup
    from pkgutil import walk_packages
    packages = [package for _, package, ispkg in walk_packages('.') if ispkg]
    metadata['requires'] = DEPENDENCIES


metadata['name'] = 'GraphEnsemble'
metadata['description'] = 'A framework for creating arbitrary graphs of machine learning models'
metadata['version'] = graph_ensemble.__version__
metadata['license'] = 'BSD 3-Clause License'
metadata['url'] = 'https://github.com/YotamFY/GraphEnsemble'
metadata['packages'] = packages

setup(**metadata)

