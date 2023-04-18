import numpy as np


class HiddenLayers(object):
    def __init__(self, layer_size: int, activation_list: list, input_list: list, output_list: list):
        self.layers = layer_size
        self.activations = activation_list
        self.inputs = input_list
        self.outputs = output_list


class MLP(object):
    def __init__(self, layers: HiddenLayers, epochs: int, learning_rate: float):
        self.layers = layers
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None

    def _param_init(self, layers, X):
        n_samples, n_features = X.shape
        self.weights = {}
        self.bias = {}
        for layer in range(layers.layers)+ 2: # +2 is because we are adding input and output layer to hidden ones
            self.weights[layer] = np.zeros(n_features)

    def forward(self, X):

        for layer in range(self.layers.layers):
            X.T @ self.weights + self.bias




layer_size = 3
activation_list = ["sigmoid", "sigmoid", "softmax"]
input_list = [100,50,10]
output_list = [50,10,5]
layers_1 = HiddenLayers(layer_size, activation_list, input_list, output_list)
