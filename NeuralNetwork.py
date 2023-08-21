import numpy as np
import math

EPSILON = 1e-10

class NeuralNetwork():
    def __init__(self, n_input, n_hlayers, n_output):
        self.n_input = n_input
        self.n_hlayers = n_hlayers
        self.n_output = n_output

        self.weight, self.biases = self.initialize_parameter()

    def initialize_parameter(self): #will initialize weight and biases to random numbers
        weight = []
        bias = []
        n_neurons = [self.n_input] + self.n_hlayers + [self.n_output]
        for i in range(1, len(n_neurons)):
            weight_matrix = np.random.randn(n_neurons[i], n_neurons[i - 1]) * np.sqrt(2 / n_neurons[i - 1])
            bias_matrix = np.zeros((n_neurons[i], 1))
            weight.append(weight_matrix)
            bias.append(bias_matrix)

        return weight, bias

    def sigmoid(self, x):
        return (1/ (1 + np.exp(-x)))
    
    def relu(self, x):
        return (np.maximum(0, x))

    def _forward_propagation(self, x):#going through each layer and calculating each layer activation value : weight * x + bias
        activation = [x]
        for i in range(len(self.weight)):
            z = np.dot(self.weight[i], activation[i]) + self.biases[i]
            a = self.relu(z)
            activation.append(a)
        
        return (activation)

    def _backward_propagation(self, x, y, activation):
        gradient = [np.zeros_like(param) for param in self.weight]
        delta = (activation[-1] - y)
    
        for i in range(len(self.weight) - 1, -1, -1):
            # print(self.weight[i].shape, activation[i].T.shape)
            gradient[i] = np.dot(delta, activation[i].T)
            delta = np.dot(self.weight[i].T, delta)
            delta *= (activation[i] > 0)

        return gradient

    def loss(self, predicted, target):
        log = 0
        if (len(predicted) != len(target)):
            raise ValueError("length missmatch: loss method")
        for i in range(len(predicted)):
            log += math.log(predicted[i] + EPSILON) * target[i]

        return (-log)

    def train(self, x_train,  y_train, epochs, learning_rate):
        for epoch in range(epochs):
            total_loss = 0

            #iterate over each training example
            for x, y in zip(x_train, y_train):
                x = x.reshape(-1, 1)
                y = y.reshape(-1, 1)

                activation = self._forward_propagation(x)
                total_loss += self.loss(activation[-1], y)
                gradients = self._backward_propagation(x, y, activation)

                for i in range(len(self.weight)):
                    self.weight[i] -= learning_rate * gradients[i]
            avg_loss = total_loss / len(x_train)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss}")

    def predict(self, x_test):
        predictions = []
        for x in x_test:
            x = x.reshape(-1, 1)
            activations = self._forward_propagation(x)
            predictions.append(activations[-1].flatten())

        return np.array(predictions)
