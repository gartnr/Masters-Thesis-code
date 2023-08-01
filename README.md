# Masters-Thesis-code
Python code for the simulations for my master's thesis: Modeling and Dynamics of Cortical Neural Networks

The code is organized as follows:
network.py defines the Network object, which holds structual information, the history, and executes the simulation of a biologically plausible neural network.
The network is comprised of model biological Neurons (three different models defined in neuron.py) and Connections between neurons (defined in connections.py).
The network can be comprised of multiple columns of neurons corresponding to the columnar organisation of the human cortex into cortical columns.
In this network model, neurons only form inter-columnar connections between nearest neighbour columns.
Column objects (defined in column.py) only serve as history repositories for column-specific data.
Support functions are defined in support.py (many of those were used for analysis and visualisation, the code for which is not included).

Multicolumn functionality of the code has not been thoroughly tested, because the analysis of single-column dynamics was broad enough for my Master's thesis.
