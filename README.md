# README
This package contains a small library of topological machine learning techniques. This package is made as part of a 
project for the Topological Machine Learning course at [UiB](https://uib.no).

## Contents
This algorithms implemented in this package can be grouped into 3 groups:

 - **Graph fitting**
   - Self organising maps
   - Growing Neural Gas
   - Reeb graphs
 - **Persistent Homology**
   - The Rips-complex algorithm
   - Column reduction for persistence diagrams
   - Persistence Images
   - Persistence Landscapes
 - **Neural Networks**
   - PersLay
   - Topological Autoencoder *(WIP: the loss function doesn't back-propagate properly yet.)*

## Documentation
The documentation for this package is hosted on [readthedocs](https://topological-machine-learning.readthedocs.io/en/latest/).

## Installing
*This package imports Pytorch and therefore requires a Python version that supports pytorch: it has been developed and 
tested using Python 3.8.*

To install this package simply use the following pip install bash command:
```bash
$ pip install TopologicalMachineLearningTechniques
```

When developing this package it is a good idea to install the package with the *dev* extras enabled. However, currently 
there are no additional dev requirements. You can install with dev dependencies using the following command:
```bash
$ pip install TopologicalMachineLearningTechniques[dev]
```

## Usage
A few Jupyter notebooks have been created to demonstrate usage of the package:
 - [Graph Fitting notebook](https://github.com/WillemSch/Topological-Learning/blob/master/src/notebooks/GraphFitting.ipynb)
 - [Persistent Homology notebook](https://github.com/WillemSch/Topological-Learning/blob/master/src/notebooks/NeuralNetworks.ipynb)
 - [Neural Network notebook](https://github.com/WillemSch/Topological-Learning/blob/master/src/notebooks/PersistentHomology.ipynb)

These notebooks can be found on the [GitHub page of this package](https://github.com/WillemSch/Topological-Learning).