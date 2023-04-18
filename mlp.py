import numpy as np


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _sigmoid_derivative(x):
    return x * (1 - x)


class MLP:
    def __init__(self, hidden_layers, epochs, learning_rate):
        self.bias_hidden_output = None
        self.weights_hidden_output = None
        self.bias_input_hidden = None
        self.weights_input_hidden = None
        self.hidden_layers = hidden_layers
        self.epochs = epochs
        self.learning_rate = learning_rate

    def fit(self, X, y):
        # Initialize weights and biases for input and hidden layers
        input_dim = X.shape[1]
        output_dim = 1  # because regression

        self.weights_input_hidden = np.random.randn(input_dim, self.hidden_layers)
        self.bias_input_hidden = np.zeros((1, self.hidden_layers))
        self.weights_hidden_output = np.random.randn(self.hidden_layers, output_dim)
        self.bias_hidden_output = np.zeros((1, output_dim))

        # Train the model for specified number of epochs
        for epoch in range(self.epochs):
            # Forward propagation
            hidden_layer_activation = np.dot(X, self.weights_input_hidden)
            hidden_layer_activation += self.bias_input_hidden
            hidden_layer_output = _sigmoid(hidden_layer_activation)

            output_layer_activation = np.dot(hidden_layer_output, self.weights_hidden_output)
            output_layer_activation += self.bias_hidden_output
            predicted_output = _sigmoid(output_layer_activation)

            # Backward propagation
            error = y - predicted_output
            d_predicted_output = error * _sigmoid_derivative(predicted_output)

            error_hidden_layer = d_predicted_output.dot(self.weights_hidden_output.T)
            d_hidden_layer = error_hidden_layer * _sigmoid_derivative(hidden_layer_output)

            # Update weights and biases
            self.weights_hidden_output += hidden_layer_output.T.dot(d_predicted_output) * self.learning_rate
            self.bias_hidden_output += np.sum(d_predicted_output, axis=0, keepdims=True) * self.learning_rate
            self.weights_input_hidden += X.T.dot(d_hidden_layer) * self.learning_rate
            self.bias_input_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * self.learning_rate

    def predict(self, X):
        # Forward propagation
        hidden_layer_activation = np.dot(X, self.weights_input_hidden)
        hidden_layer_activation += self.bias_input_hidden
        hidden_layer_output = _sigmoid(hidden_layer_activation)

        output_layer_activation = np.dot(hidden_layer_output, self.weights_hidden_output)
        output_layer_activation += self.bias_hidden_output
        predicted_output = _sigmoid(output_layer_activation)

        # Round the predicted output to 0 or 1
        predicted_output = np.round(predicted_output)
        return predicted_output
