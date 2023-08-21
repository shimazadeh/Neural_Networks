import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate
from scipy.stats import skew, kurtosis
import seaborn as sns
import sys
from NeuralNetwork import NeuralNetwork
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def save_output(arr, filename):
    with open(filename, 'w') as file:
        # Redirect the standard output to the file
        sys.stdout = file

        # Print the tabulated array to the file
        print(tabulate(arr, numalign="center"))

    sys.stdout = sys.__stdout__ 

def lab_encoder(y):
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    return (y_encoded.reshape(-1, 1))

def main():
	#read the data
    default_col_labels = [f"F{i}" for i in range(0, 32)]  # Adjust num_columns

    data = pd.read_csv("data.csv", header=None, names=default_col_labels)
    data.rename(columns={default_col_labels[0]: "id", default_col_labels[1]: "diagnosis"}, inplace=True)

    #training using NeuralNetwork
    n_input = 30  # Number of input features
    n_hlayers = [64, 32]  # Number of neurons in each hidden layer
    n_output = 1  # Number of output neurons (for binary classification)
    neural_net = NeuralNetwork(n_input, n_hlayers, n_output)

    #load the dataset
    y = lab_encoder(data["diagnosis"])
    data.drop(columns=["diagnosis", "id"], inplace=True)
    x = data.to_numpy()

    #shape check
    print(x.shape) #(num_samples, num_features)
    print(y.shape) #(num_samples, 1)
   
    #split the dataset
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Train the neural network
    epochs = 100
    learning_rate = 0.001
    neural_net.train(x_train, y_train, epochs, learning_rate)

    # Make predictions
    # predictions = neural_net.predict(x_test)

    # # Print the predictions
    # print("Predictions:")
    # print(predictions)



if __name__ == "__main__":
	main()