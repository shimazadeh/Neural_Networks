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

def main():
	#read the data
    default_col_labels = [f"F{i}" for i in range(0, 32)]

    data = pd.read_csv("data.csv", header=None, names=default_col_labels)
    data.rename(columns={default_col_labels[0]: "id", default_col_labels[1]: "diagnosis"}, inplace=True)

    #training
    n_input = 30  # Number of features
    n_hlayers = [24, 24, 24]  # Number of neurons in each hidden layer
    print("hidden layers shape", n_hlayers)
    n_output = 1  # Number of output neurons (for binary classification)
    neural_net = NeuralNetwork(n_input, n_hlayers, n_output)

    #load the dataset
    y = lab_encoder(data["diagnosis"])
    data.drop(columns=["diagnosis", "id"], inplace=True)
    x = data.to_numpy()

    min_max_scaler = MinMaxScaler()
    x_normalize = min_max_scaler.fit_transform(x)
   
    #split the dataset
    x_, x_test, y_, y_test = train_test_split(x_normalize, y, test_size=0.2, random_state=42)

    x_train, x_val, y_train, y_val = train_test_split(x_, y_, test_size=0.2, random_state=42)

    # Train the neural network
    epochs = 100
    learning_rate = 0.0001
    threshold = 0.5
    neural_net.train(x_train, y_train, x_val, y_val, epochs, learning_rate, threshold)


    #use cross validation to pick the threshold better? 

    # Make predictions
    # predictions = neural_net.predict(x_val, threshold)

    # accuracy, f1, precision, recall = neural_net.evaluate(y_val, predictions)

    # # Print the predictions
    # print(f"Precision: {precision}, accuracy: {accuracy}, f1 Score: {f1}, Recall: {recall}")

if __name__ == "__main__":
	main()