import numpy as np


class LinearRegression:
    def __init__(self, lr=0.5, n_iters=1000, penalty=None):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_pred = np.dot(X,self.weights)+self.bias

            dw = (1/n_samples) * np.dot(X.T, (y_pred-y))
            db = (1/n_samples) * np.sum(y_pred-y)

            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred


"""
class LinearRegression:
    def __init__(self, learning_rate=1e-3, n_iters=1000):
        # init parameters
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def _init_params(self):
        self.weights = np.zeros(self.n_features)
        self.bias = 0

    def _update_params(self, dw, db):
        self.weights -= self.lr * dw
        self.bias -= self.lr * db

    def _get_prediction(self, X):
        return np.dot(X, self.weights) + self.bias

    def _get_gradients(self, X, y, y_pred):
        # get distance between y_pred and y_true
        error = y_pred - y
        # compute the gradients of weight & bias
        dw = (1 / self.n_samples) * np.dot(X.T, error)
        db = (1 / self.n_samples) * np.sum(error)
        return dw, db

    def fit(self, X, y):
        # get number of samples & features
        self.n_samples, self.n_features = X.shape
        # init weights & bias
        self._init_params()

        # perform gradient descent for n iterations
        for _ in range(self.n_iters):
            # get y_prediction
            y_pred = self._get_prediction(X)
            # compute gradients
            dw, db = self._get_gradients(X, y, y_pred)
            # update weights & bias with gradients
            self._update_params(dw, db)

    def predict(self, X):
        y_pred = self._get_prediction(X)
        return y_pred
"""

"""
class LinearRegression:
    '''
    A class which implements linear regression model with gradient descent.
    '''

    def __init__(self, learning_rate=0.01, n_iterations=10000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights, self.bias = None, None
        self.loss = []

    @staticmethod
    def _mean_squared_error(y, y_hat):
        '''
        Private method, used to evaluate loss at each iteration.

        :param: y - array, true values
        :param: y_hat - array, predicted values
        :return: float
        '''
        error = 0
        for i in range(len(y)):
            error += (y[i] - y_hat[i]) ** 2
        return error / len(y)

    def fit(self, X, y):
        '''
        Used to calculate the coefficient of the linear regression model.

        :param X: array, features
        :param y: array, true values
        :return: None
        '''
        # 1. Initialize weights and bias to zeros
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        # 2. Perform gradient descent
        for i in range(self.n_iterations):
            # Line equation
            y_hat = np.dot(X, self.weights) + self.bias
            loss = self._mean_squared_error(y, y_hat)
            self.loss.append(loss)

            # Calculate derivatives
            partial_w = (1 / X.shape[0]) * (2 * np.dot(X.T, (y_hat - y)))
            partial_d = (1 / X.shape[0]) * (2 * np.sum(y_hat - y))

            # Update the coefficients
            self.weights -= self.learning_rate * partial_w
            self.bias -= self.learning_rate * partial_d

    def predict(self, X):
        '''
        Makes predictions using the line equation.

        :param X: array, features
        :return: array, predictions
        '''
        return np.dot(X, self.weights) + self.bias
"""


class MLP:
    def __init__(self, hidden_layers, epochs, learning_rate):
        self.hidden_layers = hidden_layers
        self.epochs = epochs
        self.learning_rate = learning_rate

    def fit(self, X, y):
        # Initialize weights and biases for input and hidden layers
        input_dim = X.shape[1]
        output_dim = 1 # because regression

        self.weights_input_hidden = np.random.randn(input_dim, self.hidden_layers)
        self.bias_input_hidden = np.zeros((1, self.hidden_layers))
        self.weights_hidden_output = np.random.randn(self.hidden_layers, output_dim)
        self.bias_hidden_output = np.zeros((1, output_dim))

        # Train the model for specified number of epochs
        for epoch in range(self.epochs):
            # Forward propagation
            hidden_layer_activation = np.dot(X, self.weights_input_hidden)
            hidden_layer_activation += self.bias_input_hidden
            hidden_layer_output = self._sigmoid(hidden_layer_activation)

            output_layer_activation = np.dot(hidden_layer_output, self.weights_hidden_output)
            output_layer_activation += self.bias_hidden_output
            predicted_output = self._sigmoid(output_layer_activation)

            # Backward propagation
            error = y - predicted_output
            d_predicted_output = error * self._sigmoid_derivative(predicted_output)

            error_hidden_layer = d_predicted_output.dot(self.weights_hidden_output.T)
            d_hidden_layer = error_hidden_layer * self._sigmoid_derivative(hidden_layer_output)

            # Update weights and biases
            self.weights_hidden_output += hidden_layer_output.T.dot(d_predicted_output) * self.learning_rate
            self.bias_hidden_output += np.sum(d_predicted_output, axis=0, keepdims=True) * self.learning_rate
            self.weights_input_hidden += X.T.dot(d_hidden_layer) * self.learning_rate
            self.bias_input_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * self.learning_rate

    def predict(self, X):
        # Forward propagation
        hidden_layer_activation = np.dot(X, self.weights_input_hidden)
        hidden_layer_activation += self.bias_input_hidden
        hidden_layer_output = self._sigmoid(hidden_layer_activation)

        output_layer_activation = np.dot(hidden_layer_output, self.weights_hidden_output)
        output_layer_activation += self.bias_hidden_output
        predicted_output = self._sigmoid(output_layer_activation)

        # Round the predicted output to 0 or 1
        predicted_output = np.round(predicted_output)
        return predicted_output

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoid_derivative(self, x):
        return x * (1 - x)
