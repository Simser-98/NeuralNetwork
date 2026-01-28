import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()




class LayerDense:

    def __init__(self, n_inputs, n_neurons):

        self.weights = np.random.randn(n_inputs, n_neurons) * 0.1
        self.biases = np.zeros((1, n_neurons))

        self.output = None
        self.output_error = None
        self.output_delta = None
        self.Zn_delta = None
        self.Zn_error = None

    def sigmoid(self, inputs):
        self.output =  1/(1+np.exp(-inputs))
        return self.output

    def sigmoid_derivative(self, inputs):
        self.output = self.sigmoid(inputs)*(1-self.sigmoid(inputs))

    def relu(self, inputs):
        self.output = np.maximum(0, inputs)

    def softmax(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output =  probabilities

    def feed_forward(self, inputs):
        self.output = self.sigmoid(inputs @ self.weights + self.biases)

    def backward(self, x, y, output):
        #backward propagate through the network
        self.output_error = y - output  # y is from training set, and output is network prediction
        self.output_delta = self.output_error * self.sigmoid_derivative(output)

        self.Zn_error = self.output_delta @ self.weights.T  # calculating contribution of each weight
        self.Zn_delta = self.Zn_error * self.sigmoid_derivative(self.output) # applying derivative of sigmoid to Zn error

        # update weight matrix



X,y = spiral_data(100, 3)

dense1 = LayerDense(2, 3)
activation1 = dense1.sigmoid(dense1.output)

dense2 = LayerDense(3, 3)
activation2 = dense2.sigmoid(dense2.output)

dense1.feed_forward(X)
activation1.feed_forward(dense1.output)
dense2.feed_forward(activation1.output)
activation2.feed_forward(dense2.output)

print(activation2.output)




























