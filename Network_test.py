from csv import reader
from math import exp
import pickle


#convert letters to number values
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


# Convert number values back to letters
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


# Make a prediction with a network
def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))


# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# Loads the network using pickle and runs a test based of a testfile.
def test_network():
    pkl_file = open('data.pkl', 'rb')
    network = pickle.load(pkl_file)
    pkl_file.close()
    filename = 'NeuralNet.txt'
    test = load_csv(filename)
    letter_array = []
    actual = list()
    for row in range(len(test)):
        letter_array.append(test[row][0])
        test[row].pop(0)
    letterDict = setup_letters(letter_array)
    for row in range(len(test)):
        test[row].append(letterDict[letter_array[row]])
        actual.append(test[row][-1])
    for i in range(len(test[0]) - 1):
        str_column_to_float(test, i)
    actual = list(map(int, actual))
    # convert class column to integers
    str_column_to_int(test, len(test[0]) - 1)
    # normalize input variables
    minmax = dataset_minmax(test)
    normalize_dataset(test, minmax)
    predicted = list()
    for row in test:
        prediction = predict(network, row)
        predicted.append(prediction)
    scores = accuracy_metric(actual, predicted)
    print('Scores: %s' % scores)
    predicted_letters = []
    p_letter_array = []
    actual_letters = []
    actual_array = []
    for row in range(len(predicted)):
        p_letter_array.append(predicted[row])
        actual_array.append(actual[row])
    letter_dict = put_letters(p_letter_array)
    for row in range(len(predicted)):
        predicted_letters.append(letter_dict[p_letter_array[row]])
        actual_letters.append(letter_dict[actual_array[row]])
    print(''.join(predicted_letters))
    print(''.join(actual_letters))


def main():
    test_network()


if __name__ == "__main__":
     main()

