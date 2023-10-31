import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tabulate import tabulate
from scipy.stats import skew, kurtosis
import seaborn as sns
import sys
from NeuralNetwork import NeuralNetwork
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def save_output(arr, filename):
    with open(filename, 'w') as file:
        sys.stdout = file

        print(tabulate(arr, numalign="center"))
    sys.stdout = sys.__stdout__ 

def lab_encoder(y):
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    return (y_encoded.reshape(-1, 1))

def random_search(x_train, y_train, x_val, y_val):
    n_input = 30
    n_output = 1
    epochs = 100
    threshold = 0.5

    n_hlayers = [[16, 8], [32, 16], [32, 16, 8]]
    learning_rate = [0.001, 0.0001, 0.00001]
    output = []

    for i in range(3):
        for j in range(3):
            print(f"Training with {n_hlayers[i]} layers and LR={learning_rate[j]}")

            neural_net = NeuralNetwork(n_input, n_hlayers[i], n_output)
            accuracy, val_acc = neural_net.train(x_train, y_train, x_val, y_val, epochs, learning_rate[j], threshold)
            output.append([n_hlayers[i], learning_rate[j], accuracy, val_acc])

    print(pd.DataFrame(output, columns=["n layers", "LR", "Accuracy", "Validation Accuracy"]))

def main():
	#read and load the data
    default_col_labels = ["id", "diagnosis"]
    for i in range(1, 31):
        default_col_labels.append(f"F{i}")
    data = pd.read_csv("data.csv", header=None, names=default_col_labels)

    #cleanup the dataset
    y = lab_encoder(data["diagnosis"])
    x = data.iloc[:,2:]
    min_max_scaler = MinMaxScaler()
    x_normalize = min_max_scaler.fit_transform(x)

    #split the dataset
    x_, x_test, y_, y_test = train_test_split(x_normalize, y, test_size=0.2, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_, y_, test_size=0.2, random_state=42)

    # model architechture
    n_input = 30
    n_hlayers = [32, 16]
    n_output = 1  
    epochs = 100
    threshold = 0.5
    learning_rate = 0.03652

    neural_net = NeuralNetwork(n_input, n_hlayers, n_output)
    accuracy, val_acc = neural_net.train(x_train, y_train, x_val, y_val, epochs, learning_rate, threshold)

    #use cross validation to pick the threshold better? 

    # Make predictions
    predictions = neural_net.predict(x_test, threshold)

    accuracy, f1, precision, recall = neural_net.evaluate(y_test, predictions)

    # # Print the predictions
    print(f"Precision: {precision}, accuracy: {accuracy}, f1 Score: {f1}, Recall: {recall}")

if __name__ == "__main__":
	main()