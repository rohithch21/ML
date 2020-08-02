''' Neural Network from Scratch with back prop '''
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 


def load_csv_dataset():
    dataset = pd.read_csv("seed_dataset.csv")
    return dataset

dataset = load_csv_dataset()
X,y = dataset.iloc[:,0:7],dataset.iloc[:,7]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
X_train_cols = X_train.shape()[1]

nn_architecture = [
    {"input_dim": X_train_cols, "output_dim": 6, "activation": "relu"},
    {"input_dim": 6, "output_dim": 3, "activation": "Sigmoid"},
]

def init_layers(nn_architecture, seed = 99):
    np.random.seed(seed)
    number_of_layers = len(nn_architecture)
    params_values = {}

    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
        layer_input_size = layer["input_dim"]
        layer_output_size = layer["output_dim"]
        
        params_values['W' + str(layer_idx)] = np.random.randn(
            layer_output_size, layer_input_size) * 0.1
        params_values['b' + str(layer_idx)] = np.random.randn(
            layer_output_size, 1) * 0.1

def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def relu(Z):
    return np.maximum(0,Z)

def sigmoid_backward(dA, Z):
    sig = sigmoid(Z)
    return dA * sig * (1 - sig)

def relu_backward(dA, Z):
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0
    return dZ

# def layer_to_layer_op(nn_architecture,W_curr,bias_curr):
#     # Z(L) = W(L,L+1) * a(L-1) + b(L)   Lth - layer (input layer = 1)
#     a_prev = 
#     Z = np.dot(W_curr,a_prev) + bias_curr

# def full_feed_forward(X_train):
#     A_curr = X_train
    
#     for idx,layer in enumerate(nn_architecture):
#         layer_idx = idx + 1
#         num_layer_units_left = layer["input_dim"]
#         num_layer_units_right = layer["output_dim"]
        


