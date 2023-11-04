# Neural Network From Scratch 
## Description
This program construct a neural network that can be used with different options for binary classification.

## Prerequisites
- Python 3.6 or higher
- Required libraries: pandas, numpy, matplotlib, sklearn, scipy, seaborn, optuna

## Getting Started
1. Clone this repository to your local machine.
2. Place your dataset in a file named `data.csv`. The dataset should have two columns: `id` and `diagnosis`, where `diagnosis` contains the labels, and the remaining columns should contain the features for each sample.

## Usage
### Training and Optimization
There are two mode to this program: 1. trial 2.model. If you already know the parameters you want to use you can use option 2 but if you dont know the best parameters that should be used for your data set you can use trial mode. ensure you adjust the "Objective" method in the main.py to adjust the range of the parameters you want to study: 


    - Command to execute the optimization process using Optuna to find the best hyperparameters for the network: python main.py --mode trial: 
    - Command to construct the model with your desired values: python main.py --mode model --layers <layers> --epochs <epochs> --learning_rate <learning_rate>

## Results
For this dataset we studied the data before processing them. see the result in DataAnalysis.ipynb

The training results, including loss curves, accuracy, and evaluation metrics, will be displayed in the terminal during terminal and the associated graph is saved as .png file.

