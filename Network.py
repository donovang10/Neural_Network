from random import randrange
from random import random
from csv import reader
from math import exp
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import pickle


# Load a CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


# creates a dictionary to change letters to numbers
def setup_letters(letters_input):
    letters = dict.fromkeys(letters_input)
    letters['A'] = 0
    letters['B'] = 1
    letters['C'] = 2
    letters['D'] = 3
    letters['E'] = 4
    letters['F'] = 5
    letters['G'] = 6
    letters['H'] = 7
    letters['I'] = 8
    letters['J'] = 9
    letters['K'] = 10
    letters['L'] = 11
    letters['M'] = 12
    letters['N'] = 13
    letters['O'] = 14
    letters['P'] = 15
    letters['Q'] = 16
    letters['R'] = 17
    letters['S'] = 18
    letters['T'] = 19
    letters['U'] = 20
    letters['V'] = 21
    letters['W'] = 22
    letters['X'] = 23
    letters['Y'] = 24
    letters['Z'] = 25
    return letters


# Dictionary of numbers to letters
def put_letters(letters_input):
    letters = dict.fromkeys(letters_input)
    letters[0] = 'A'
    letters[1] = 'B'
    letters[2] = 'C'
    letters[3] = 'D'
    letters[4] = 'E'
    letters[5] = 'F'
    letters[6] = 'G'
    letters[7] = 'H'
    letters[8] = 'I'
    letters[9] = 'J'
    letters[10] = 'K'
    letters[11] = 'L'
    letters[12] = 'M'
    letters[13] = 'N'
    letters[14] = 'O'
    letters[15] = 'P'
    letters[16] = 'Q'
    letters[17] = 'R'
    letters[18] = 'S'
    letters[19] = 'T'
    letters[20] = 'U'
    letters[21] = 'V'
    letters[22] = 'W'
    letters[23] = 'X'
    letters[24] = 'Y'
    letters[25] = 'Z'
    return letters


# Convert string column to float
def str_column_to_float(dataset, column):

    for row in dataset:
        row[column] = float(row[column].strip())


# Convert string column to integer
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup


# Find the min and max values for each column
def dataset_minmax(dataset):
    minmax = list()
    stats = [[min(column), max(column)] for column in zip(*dataset)]
    return stats


# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row) - 1):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])


# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        predicted_letters = []
        actual_letters = []
        p_letter_array = []
        a_letter_array = []
        labels = ["A", "B", "C", "D", "E", "F", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U",
                  "V", "W", "X", "Y", "Z"]
        for row in range(len(predicted)):
            p_letter_array.append(predicted[row])
        letter_dict = put_letters(p_letter_array)
        for row in range(len(predicted)):
            predicted_letters.append(letter_dict[p_letter_array[row]])
        for row in range(len(actual)):
            a_letter_array.append(actual[row])
        letter_dict = put_letters(a_letter_array)
        for row in range(len(actual)):
            actual_letters.append(letter_dict[a_letter_array[row]])
        #cm_analysis(actual_letters, predicted_letters, 'Confusion/confusion_matrix5_.4_100_26.png', labels)
        scores.append(accuracy)
    return scores


# Evaluate an algorithm using a cross validation split
def num_evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        #cm_analysis(actual, predicted, 'Confusion/confusion_matrix5_.4_100_26.png', labels)
        scores.append(accuracy)
    return scores


# Calculate neuron activation for an input
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return activation


# Transfer neuron activation
def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))


# Forward propagate input to a network output
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


# Calculate the derivative of an neuron output
def transfer_derivative(output):
    return output * (1.0 - output)


# Back propagate error and store in neurons
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])


# Update network weights with error
def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta']


# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        for row in train:
            forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)


# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights': [random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights': [random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network


# Make a prediction with a network
def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))


# Back propagation Algorithm With Stochastic Gradient Descent
def back_propagation(train, test, l_rate, n_epoch, n_hidden):
    n_inputs = len(train[0]) - 1
    n_outputs = len(set([row[-1] for row in train]))
    network = initialize_network(n_inputs, n_hidden, n_outputs)
    train_network(network, train, l_rate, n_epoch, n_outputs)
    output = open('data.pkl', 'wb')
    # Pickle dictionary using protocol 0.
    pickle.dump(network, output)
    pickle.dump(test, output, -1)
    output.close()
    print(network)
    predictions = list()
    for row in test:
        prediction = predict(network, row)
        predictions.append(prediction)
    return predictions


# creates an naccuracy graph for the overall accuracy of each fold
def accuracy_graph(scores):
    plt.plot(scores)
    plt.title('model accuracy chart 1')
    plt.ylabel('accuracy')
    plt.xlabel('fold')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('Accuracy/accuracy_chart5_.4_100_26.png')


def cm_analysis(y_true, y_pred, filename, labels, ymap=None, figsize=(26, 26)):
    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt='', ax=ax)
    plt.savefig(filename)


# runs through the network using a dataset for handwritten letters
def letters_run():
    # load and prepare data
    filename = 'letters.csv'
    dataset = load_csv(filename)
    letter_array = []
    for row in range(len(dataset)):
        letter_array.append(dataset[row][0])
        dataset[row].pop(0)
    letterDict = setup_letters(letter_array)
    for row in range(len(dataset)):
        dataset[row].append(letterDict[letter_array[row]])
    for i in range(len(dataset[0]) - 1):
        str_column_to_float(dataset, i)
    # convert class column to integers
    str_column_to_int(dataset, len(dataset[0]) - 1)
    # normalize input variables
    minmax = dataset_minmax(dataset)
    normalize_dataset(dataset, minmax)
    # evaluate algorithm
    n_folds = 5
    l_rate = 0.6
    n_epoch = 100
    n_hidden = 26
    scores = evaluate_algorithm(dataset, back_propagation, n_folds, l_rate, n_epoch, n_hidden)
    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))
    # accuracy_graph(scores)


# runs a network against a number dataset. Still need to find a good one.
def numbers_run():
    # load and prepare data
    filename = 'train.csv'
    dataset = load_csv(filename)
    dataset.pop(0)
    for i in range(len(dataset[0]) - 1):
        str_column_to_float(dataset, i)
    # convert class column to integers
    str_column_to_int(dataset, len(dataset[0]) - 1)
    # evaluate algorithm
    n_folds = 4
    l_rate = 0.4
    n_epoch = 100
    n_hidden = 26
    scores = num_evaluate_algorithm(dataset, back_propagation, n_folds, l_rate, n_epoch, n_hidden)
    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))
    num_accuracy_graph(scores)


def main():
    letters_run()
    #numbers_run()


if __name__ == "__main__":
    main()
