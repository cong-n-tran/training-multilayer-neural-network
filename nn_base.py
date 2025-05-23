# Cong Tran
# 1002046419

import numpy as np
from uci_load import read_uci_file
from nn_solution import nn_train_and_test

class hyperparameters:
    num_layers = None       # should be an integer >= 2
    units_per_layer = None  # should be a list of the number of units in the hidden layers.
                            # The length of this list should be num_layers - 2.
    training_rounds = None  # specifies the number of training rounds.


# Feel free to change the default directory to point to where you have
# stored the datasets.
directory = "uci_datasets"

# Feel free to change the dataset name, so that you can experiment
# with different datasets.
# dataset = "pendigits"
# dataset = "satellite"
dataset = "yeast"

# Feel free to change the values in the next lines, so that you can experiment
# with different parameters.
parameters = hyperparameters()

# default
parameters.num_layers = 4
parameters.units_per_layer = [20, 15]
parameters.training_rounds = 20

#config 1 - i get the yeast to be 0.3017 everytime as what the professor expects
# parameters.num_layers = 6
# parameters.units_per_layer = [20, 30, 25, 30]
# parameters.training_rounds = 20

# #config 2 ---- the best rsult as of right now -> faster and the same training round / even less
# parameters.num_layers = 3
# parameters.units_per_layer = [30]
# parameters.training_rounds = 20


training_file = directory + "/" + dataset + "_training.txt"
test_file = directory + "/" + dataset + "_test.txt"

labels_to_ints = {}
ints_to_labels = {}

# These lines read the dataset
(tr_data, tr_labels) = read_uci_file(training_file, labels_to_ints, ints_to_labels)
(test_data, test_labels) = read_uci_file(test_file, labels_to_ints, ints_to_labels)

# This is where your code is called.
nn_train_and_test(tr_data, tr_labels, test_data, test_labels,
                  labels_to_ints, ints_to_labels, parameters)
