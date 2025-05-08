# Cong Tran
# 1002046419
import numpy as np

# parameters is an object of class hyperparameters, defined in nn_base.py, that includes the following variables:
# num_layers: specifies the number of layers in the network. 
#   The number of layers cannot be smaller than 2, 
#   since any neural network should have at least an input layer and an output layer.
# units_per_layer: should be a list of the number of units in the hidden layers. 
#   The length of this list should be num_layers - 2. 
#   For example, units_per_layer[0] is the number of units in the first hidden layer (assuming num_layers >= 3), 
#   units_per_layer[1] is the number of units in the second hidden layer (assuming num_layers >= 4), and so on.
# training_rounds: 
#   specifies the number of training rounds, as in Task 1 and Task 2.

def nn_train_and_test(tr_data: any , tr_labels: any, test_data: any, test_labels: any,
                  labels_to_ints: dict, ints_to_labels: dict, parameters: any) -> None: 
    
    num_layers  = parameters.num_layers # a single number -> would be num_layer - 2 = number of hidden layers
    units_per_layer = parameters.units_per_layer # an array where index corresponds to the number of units in a particular hidden layer
    training_rounds = parameters.training_rounds # single number -> same as before
    # print(f'this is the number of layers: {num_layers}')
    # print(f'this is the number of units per layer: {units_per_layer}')
    # print(f'this is the number of training rounds: {training_rounds}')

    # same as before
    absolute_max_value = np.max(np.abs(tr_data))
 

    normalized_training_data = tr_data / absolute_max_value
    normalized_test_data = test_data / absolute_max_value

    input_array_length, input_dimensions = normalized_training_data.shape 

    num_classes = len(labels_to_ints) # this is for the conversion to one hot vectors
    num_layers = len(units_per_layer) + 1 # now we need to iterate through each layer

    # returns the arrays of weights and biases 
    weights, biases = initialize_weights(units_per_layer=units_per_layer, input_dimensions=input_dimensions, num_classes=num_classes)

    # initial_loss = loss_function(w=weights, b=biases, training_data=normalized_training_data, training_label= tr_labels, length= input_array_length, num_classes=num_classes, num_layers=num_layers)
    # print(f' this is the current loss of the function {initial_loss}')

    for round in range(1, training_rounds + 1): 
        # learning rate update after eery round
        learning_rate  = 0.98 ** round

        for i in range(input_array_length): 
            
            #initalize the values 
            input_vector = normalized_training_data[i].reshape(-1, 1)
            output_label = tr_labels[i, 0]
            one_hot_vector_output = create_one_hot_vector(output_label, num_classes)
            
            # forward pass

            # to get the z value of each layer
            z_array = []
            z = input_vector

            for j in range(num_layers): 
                a = pre_activation_function(weights[j], biases[j], z)
                z = activation_function(a)
                z_array.append(z)

            # backward propagation

            # get the deltas of each layer 
            deltas = backward_propagation(w=weights, z_array=z_array, num_layers=num_layers, one_hot_vector_output=one_hot_vector_output, deltas=[None] * num_layers, layer=0)

            #updating the weights
            curr_z = input_vector
            for k in range(num_layers): 
                dldw = np.dot(curr_z, deltas[k].T)
                dldb = deltas[k].T
                curr_z = z_array[k]
                
                weights[k] -= learning_rate * dldw
                biases[k] -= learning_rate * dldb

    # initial_loss = loss_function(w=weights, b=biases, training_data=normalized_training_data, training_label= tr_labels, length= input_array_length, num_classes=num_classes, num_layers=num_layers)
    # print(f' this is the current loss of the function {initial_loss}')


    # the evaluation

    correct = 0
    test_array_length, test_dimensions = normalized_test_data.shape 
    
    for i in range(test_array_length): 

        #initalize the values 
        input_test_vector = normalized_test_data[i].reshape(-1, 1)
        output_test_label = test_labels[i, 0]

        #calculate the values
        z = input_test_vector
        for j in range(num_layers): 
            a = pre_activation_function(weights[j], biases[j], z)
            z = activation_function(a)
        predicted = ints_to_labels[np.argmax(z)]
        true_class = ints_to_labels[output_test_label]
        accuracy = test_output(z, output_test_label)
        correct += accuracy

        print_test_object(object_id=i, predicted_class=predicted, true_class=true_class, accuracy=accuracy)

    classification_accuracy = correct / test_array_length
    print_classification_accuracy(classification_accuracy=classification_accuracy)

    

def backward_propagation(w: any, z_array: int, num_layers: 
                         int, one_hot_vector_output: int, deltas, layer: int, ) -> tuple: 
    
    if layer == num_layers - 1:  #base case of when we are at the output layer
        
        #create the one hot vector to produce the error
        one_hot_vector_output = np.array(one_hot_vector_output).reshape(-1, 1)

        # this is a vector of the output layer of error --> (z[l] - tn) * z[l] * (1 - z[l])
        output_delta = z_array[layer] * (1 - z_array[layer]) * (z_array[layer] - one_hot_vector_output)

        # put it at the end of the deltas array (representation of it being at the output layer)
        deltas[layer] = output_delta
        return deltas

    # gets the array of deltas of the layer in front of it
    future_delta = backward_propagation(w=w, z_array=z_array, num_layers=num_layers, 
                                        one_hot_vector_output=one_hot_vector_output, deltas=deltas, layer=layer + 1)

    #formula from slide 43 -> (d[l + 1] *  w[l + 1]) * z[l] * (1 - z[l])
    future_error = np.dot(w[layer + 1], future_delta[layer + 1])
    dzda = z_array[layer] * (1 - z_array[layer])
    deltas[layer] = future_error * dzda

    # will return an array of detlas that are filled up with the error of each layer based on the next layer 
    return deltas 
    

#partial derivative of error function with respect to w - NOT USED
def partial_derivative_w(w: any, b: any, tn: any, xn: any, num_classes: int) -> float:
    a = pre_activation_function(w=w, b=b, x=xn)
    z = activation_function(a)
    tn_one_hot_vector = create_one_hot_vector(position=tn, num_class=num_classes)

    # (z(n) - tn) * (1 - z(xn)) * z(xn) * xn
    error = (z - tn_one_hot_vector)
    dzda = z * (1 - z) 
    dldw = np.dot(xn, (error * dzda))
    return dldw

# partial derivative of error function with respect to b - NOT USED
def partial_derivative_b(w: float, b: float, tn: any, xn: any, num_classes: int) -> float:
    a = pre_activation_function(w=w, b=b, x=xn)
    z = activation_function(a)
    tn_one_hot_vector = create_one_hot_vector(position=tn, num_class=num_classes)

    # (z(xn) - tn) * z(xn) * (1 - z(xn))
    error = (z - tn_one_hot_vector)
    dzda = z * (1 - z)
    dldb = dzda * error
    return dldb


# calculate the error of the entire network
def loss_function(w: any, b: any, training_data: any, training_label: any, length: int, num_classes: int, num_layers: int) -> float:

    loss = 0
    for i in range(length): 
        input_vector = training_data[i].reshape(-1, 1)
        output_label = training_label[i]
        one_hot_vector_output = create_one_hot_vector(output_label, num_classes)
        
        #going through all the layers
        z = input_vector
        for j in range(num_layers): 
            a = pre_activation_function(w[j], b[j], z)
            z = activation_function(a)
        square_differences = 0.5 * ((z - one_hot_vector_output) ** 2)
        loss += np.sum(square_differences)
    return loss

# create the one hot vector 
def create_one_hot_vector(position: int, num_class: int) -> any: 
    one_hot_vector = np.zeros(num_class)
    one_hot_vector[position] = 1
    return one_hot_vector

def pre_activation_function(w: any, b: any, x: any) -> any: 
    w_T = np.transpose(w)
    b_T = np.transpose(b)
    return np.dot(w_T, x) + b_T

# this is basically e^(-b - wT @ xn)
def e_function(a: float): 
    return np.exp(- a)

def activation_function(a: float): 
    return 1 / (1 + e_function(a))

def initialize_weights(units_per_layer: list, input_dimensions, num_classes: int)-> tuple: 
    # if there are only 2 layers i.e input and output layer
    if len(units_per_layer) == 0: 
        weights = [get_random_weights(input_dimensions, num_classes)]
        biases = [get_random_weights(1, num_classes)]
        return weights, biases

    #initalized weights and biases
    weights = []
    biases = []

    #input layer -> first hidden layer
    weights.append(get_random_weights(input_dimensions, units_per_layer[0]))
    biases.append(get_random_weights(1, units_per_layer[0]))
    for i in range(len(units_per_layer)-1): 
        
        #ith hidden layer -> ith + 1 hidden layer
        weights.append(get_random_weights(units_per_layer[i], units_per_layer[i+1]))
        biases.append(get_random_weights(1, units_per_layer[i + 1]))

    #last hidden layer -> output layer
    weights.append(get_random_weights(units_per_layer[-1], num_classes))
    biases.append(get_random_weights(1, num_classes))

    return weights, biases
        


def get_random_weights(row: int, col: int) -> any: 
    return np.random.uniform(-0.5, 0.5, size = (row, col))


# print test objects
def print_test_object(object_id: int, predicted_class: any, true_class: any, accuracy: float): 
    print('ID=%5d, predicted=%10s, true=%10s, accuracy=%4.2f\n' % 
           (object_id, str(predicted_class), str(true_class), accuracy))
    
# print classification accuracy
def print_classification_accuracy(classification_accuracy: float) -> None: 
    print('classification accuracy=%6.4f\n' % (classification_accuracy))


def test_output(predicted_scores: any, correct_class: int) -> float: 
    predicted_scores = np.squeeze(predicted_scores) 

    maximum_score = np.max(predicted_scores)

    tied_classes = np.where(predicted_scores == maximum_score)[0]

    # if we only get one maximum value then check if its correct of not
    if len(tied_classes) == 1: 
        predicted_class = tied_classes[0]
        if predicted_class == correct_class: 
            return 1
    # we got ties 
    else: 
        # if the correct class is withiin the ties -> divided by the number of ties
        if correct_class in tied_classes: 
            return 1.0 / len(tied_classes)

    # all ifs fail -> it is wrong then 
    return 0