import joblib
import panda as pd
import numpy as np
import sys

def main():
    if (sys.argv[0] is None):
        return -1 , print("No model is provided")
    
    loaded_neural_net = joblib.load(sys.argv[0])
    if (loaded_neural_net is None):
        return -1, print("Error loading the model")

    predictions = loaded_neural_net.predict(x_test, threshold)

if __name__ == "__main__":
    main()