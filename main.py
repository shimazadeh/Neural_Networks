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
from Model.NeuralNetwork import NeuralNetwork
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import optuna
import argparse

def save_output(arr, filename):
    with open(filename, 'w') as file:
        sys.stdout = file

        print(tabulate(arr, numalign="center"))
    sys.stdout = sys.__stdout__ 

def lab_encoder(y):
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    return (y_encoded.reshape(-1, 1))

def training(layers, epochs, learning_rate, l1=0, l2=0):
    #read and load the data
    default_col_labels = ["id", "diagnosis"]
    for i in range(1, 31):
        default_col_labels.append(f"F{i}")
    data = pd.read_csv("./Data/data.csv", header=None, names=default_col_labels)

    #cleanup/scale the dataset
    y = lab_encoder(data["diagnosis"])
    x = data.iloc[:,2:]
    min_max_scaler = MinMaxScaler()
    x_normalize = min_max_scaler.fit_transform(x)

    #split the dataset
    x_, x_test, y_, y_test = train_test_split(x_normalize, y, test_size=0.2, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_, y_, test_size=0.2, random_state=42)

    #model architechture
    n_input = 30
    n_output = 1  
    threshold = 0.5
    neural_net = NeuralNetwork(n_input, layers, n_output, l1, l2)
    accuracy, val_acc = neural_net.train(x_train, y_train, x_val, y_val, int(epochs), learning_rate, threshold)

    # Make predictions
    predictions = neural_net.predict(x_test, threshold)
    accuracy, f1, precision, recall = neural_net.evaluate(y_test, predictions)

    # Print the predictions
    print(f"Precision: {precision}, accuracy: {accuracy}, f1 Score: {f1}, Recall: {recall}")
    return accuracy

def objective(trial):
    n1_layers = trial.suggest_categorical('n1_layers', [128, 64, 32, 16, 8])
    n2_layers = trial.suggest_categorical('n2_layers', [32, 16, 8])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1)
    l1 = trial.suggest_float('l1', 1e-6, 1e-3)
    l2 = trial.suggest_float('l2', 1e-6, 1e-3)

    acc = training(n1_layers=n1_layers, n2_layers=n2_layers, LR=learning_rate, l1=l1, l2=l2)
    metric =np.mean(acc)

    return metric  
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str)
    parser.add_argument("--layers", type=int, nargs='+')
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning_rate", type=float)
    # parser.add_argument("--batch_size", type=str)

    arg = parser.parse_args()
    
    mode = arg.mode
    epochs = arg.epochs
    learning_rate = arg.learning_rate
    # batch_size = arg.batch_size
    layers = arg.layers

    if mode == "trial":
        study = optuna.create_study(direction='maximize', storage="sqlite:///db.sqlite3", study_name="NN-3")
        study.optimize(objective, n_trials=50)
    elif mode == "model":
        training(layers, epochs, learning_rate)
    else:
        print("Mode is not provided")

###the best trial for this data is 
#(64,32), learning_rate = 0.0137076298935458, l1=8.16086922773294e-06, l2=1.45963292232904e-06
#parameters: {'n1_layers': 32, 'n2_layers': 32, 'learning_rate': 0.09091346388064911, 'l1': 3.715254818748309e-05, 'l2': 0.00011241721456937402}