import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns

EPSILON = 1e-10
np.random.seed(42) 

class NeuralNetwork():
    def __init__(self, n_input, n_hlayers, n_output):
        self.n_input = n_input
        self.n_hlayers = n_hlayers
        self.n_output = n_output

        self.losses = []
        self.accuracies = []
        self.validation_losses = []
        self.validation_accuracies = []

        self.learning_rate = 0
        self.epochs = 0
        self.weight, self.biases = self.initialize_parameter()

    def initialize_parameter(self):
        weight = []
        bias = []
        n_neurons = [self.n_input] + self.n_hlayers + [self.n_output]
        for i in range(1, len(n_neurons)):
            weight_matrix = np.random.randn(n_neurons[i], n_neurons[i - 1]) * np.sqrt(2 / n_neurons[i - 1])
            bias_matrix = 0.001 * np.ones((n_neurons[i], 1))
            weight.append(weight_matrix)
            bias.append(bias_matrix)

        return weight, bias

    def sigmoid(self, x):
        return (1/ (1 + np.exp(-x)))
    
    def relu(self, x):
        return (np.maximum(0, x))

    def _forward_propagation(self, x):#going through each layer and calculating each layer activation value : weight * x + bias
        activation = [self.sigmoid(x)]
        for i in range(len(self.weight)):
            z = np.dot(self.weight[i], activation[i]) + self.biases[i]
            a = self.relu(z)
            activation.append(a)
        return (activation)

    def _backward_propagation(self, x, y, activation):
        gradient = [np.zeros_like(param) for param in self.weight]
        delta = (activation[-1] - y)
    
        for i in range(len(self.weight) - 1, -1, -1):
            gradient[i] = np.dot(delta, activation[i].T)
            delta = np.dot(self.weight[i].T, delta)
            delta *= (activation[i] > 0)
        return gradient

    def loss(self, predicted, target):
        if (len(predicted) != len(target)):
            raise ValueError("length missmatch: loss method")
        
        log = 0
        for i in range(len(predicted)):
            temp = max(min(predicted[i], 1 - EPSILON), EPSILON)
            log += -target[i] * math.log(temp) - ((1 - target[i]) * math.log(1- temp))
            
        #L1 regularization
        L1_regularization = 0
        for layer_weights in self.weight:
            L1_regularization += np.sum(np.abs(layer_weights))
        
        #L2 regularization
        L2_regularization = 0
        for layer_weights in self.weight:
            L2_regularization += np.sum((layer_weights)**2)

        regularized_log = (log) + 0.001 * L2_regularization
        # print(L1_regularization, L2_regularization, log)
        return (log)

    def train(self, x_train,  y_train, x_val, y_val, _epochs, _learning_rate, threshold):
        self.learning_rate = _learning_rate
        self.epochs = _epochs
        
        for _e in range(self.epochs):
            total_loss = 0

            #training
            for x, y in zip(x_train, y_train):
                x = x.reshape(-1, 1)
                y = y.reshape(-1, 1)
                
                activation = self._forward_propagation(x)
                total_loss += self.loss(activation[-1], y)
                gradients = self._backward_propagation(x, y, activation)
               
                for i in range(len(self.weight)):
                    self.weight[i] -= self.learning_rate * gradients[i]
            
            avg_loss = total_loss / len(x_train)
            self.losses.append(avg_loss)
            prediction = self.predict(x_train, threshold)
            accuracy = accuracy_score(prediction, y_train)
            self.accuracies.append(accuracy)

            #validation
            val_total_loss = 0
            val_correct_predictions = 0
            for x, y in zip(x_val, y_val):
                x = x.reshape(-1, 1)
                y = y.reshape(-1, 1)
                val_activation = self._forward_propagation(x)
                val_total_loss += self.loss(val_activation[-1], y)
                val_predictions = self.predict([x], threshold)
                val_correct_predictions += (val_predictions == y).all()

            avg_val_loss = val_total_loss / len(x_val)
            val_accuracy = val_correct_predictions / len(x_val)
            self.validation_losses.append(avg_val_loss)
            self.validation_accuracies.append(val_accuracy)
        
            print(f"Epoch {_e+1}/{_e}, Loss: {avg_loss}, accuracy: {accuracy}, val_loss: {avg_val_loss}, val_acc: {val_accuracy}")
        
        self._plot(self.losses, self.validation_losses, "Loss", "Loss_figure.png")
        self._plot(self.accuracies, self.validation_accuracies, "Accuracies", "Learning_curve.png")
    
        return self.accuracies[-1], self.validation_accuracies[-1]
    
    def predict(self, x_test, threshold):
        probabilities = []
        for x in x_test:
            x = x.reshape(-1, 1)
            activations = self._forward_propagation(x)
            probabilities.append(activations[-1].flatten())
        
        predictions = [1 if prob >= threshold else 0 for prob in probabilities]
        return np.array(predictions)
    
    def _plot(self, data1, data2, ylabel, filename):
        plt.figure()
        plt.plot(range(len(data1)), data1, label="Training", color="blue")
        plt.plot(range(len(data2)), data2, label="Validation", linestyle="-.", color="magenta")
        plt.xlabel("Epoch iteration")
        plt.ylabel(ylabel)
        plt.legend()
        plt.savefig(filename)

    def evaluate(self, y, y_pred):
        cm = confusion_matrix(y, y_pred)
        plt.figure(figsize=(8,6))
        sns.heatmap(cm)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig("ConfusionMatrix.png")
        
        accuracy = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)

        return (accuracy, f1, precision, recall)