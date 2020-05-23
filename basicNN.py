import numpy as np

def sigmoid(x):
    # Activation function: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs) :
        # Weight inputs, add bias, then use the activation function
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)

weights = np.array([0, 1])
bias = 4
n = Neuron(weights, bias)

x = np.array([2, 3])
print(n.feedforward(x))

class ourNeuralNetwork :
    '''

    This neural network has:
    - 2 inputs
    - 1 hidden layer with 2 neurons (h1, h2)
    - an output layer with only 1 neuron (01)
    Each neuron has the same weight and bias:
    - w = [0, 1]
    - b = 0
    '''
    def __init__(self):
        weights = np.array([0, 1])
        bias = 0
    
        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights, bias)
        self.o1 = Neuron(weights, bias)

    def feedforward(self, x) :
        out_h1 = self.h1.feedforward(x)
        out_h2 = self.h2.feedforward(x)

        out_01 = self.o1.feedforward(np.array([out_h1, out_h2]))

        return out_01

network = ourNeuralNetwork()
x = np.array([2, 3])
print(network.feedforward(x))