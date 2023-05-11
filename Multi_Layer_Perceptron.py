import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import time

from linear_models import r_squared_score
import data_preprocess
import testing_model

np.random.seed(6)


# Defining sigmoid activation function
def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Defining derivative of  sigmoid activation function
def _sigmoid_derivative(x):
    return _sigmoid(x) * (1 - _sigmoid(x))


# Defining mean squared error (MSE) loss function
def _mse(y_true, y_predictions):
    return np.mean((y_true - y_predictions) ** 2)


# Defining the derivative of mean squared error (MSE) loss function
def _mse_derivative(y_true, y_predictions):
    return -2 * (y_true - y_predictions)


# Defining root mean squared error (RMSE) loss function
def _rmse(y_true, y_predictions):
    return np.sqrt(np.mean((y_true - y_predictions) ** 2))


# Defining the derivative of the root mean squared error (RMSE) loss function
def _rmse_derivative(y_true, y_predictions):
    return -1 * (y_true - y_predictions) / _rmse(y_true, y_predictions)


# Defining the ReLU activation function
def _relu(x):
    return np.maximum(0, x)


# Defining the derivative of the ReLU activation function
def _relu_derivative(x):
    return (x > 0).astype(float)


# Defining the Multi Layer Perceptron Regressor ( MLPRegressor) class
class MLPRegressor:
    def __init__(self, hidden_layer_sizes=(100,), learning_rate=0.01, max_iter=1000, random_state=None, beta1=0.9,
                 beta2=0.999, epsilon=1e-8, activation="relu", initialization="random", loss_func="mse"):
        # Initialization of hidden layers and hyperparameters
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.max_iter = max_iter

        # Assigning a random state and creating empty lists for weights and biases
        self.random_state = random_state
        self.weights = []
        self.biases = []
        self.activation = activation
        self.loss_func = loss_func
        self.initialization = initialization

        # Initialization of Adam Optimizer parameters
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_t = None
        self.v_t = None

        # Creating lists for storing the values for validation scores and training losses
        self.val_scores = []
        self.loss_values = []

        # Creating lists for storing the values for running val scores and train losses
        self.running_avg_val_scores = []
        self.running_avg_loss_values = []

    # This function initializes the weights and biases of the MLP.
    # The weights and biases are initialized according to the specified initialization method
    def _initialize_parameters(self, layer_sizes):
        if self.random_state:
            np.random.seed(self.random_state)

        for i in range(len(layer_sizes) - 1):
            input_units = layer_sizes[i]
            output_units = layer_sizes[i + 1]

            # Initialize weights and biases with Zeros initialization
            if self.initialization == 'zeros':
                self.weights.append(np.zeros((input_units, output_units)))
                self.biases.append(np.zeros(output_units))
            # Initialize weights and biases with Ones initialization
            elif self.initialization == 'ones':
                self.weights.append(np.ones((input_units, output_units)))
                self.biases.append(np.ones(output_units))
            # Initialize weights and biases with Xavier initialization
            elif self.initialization == 'xavier':
                std_dev = np.sqrt(1 / input_units)
                self.weights.append(np.random.normal(0, std_dev, (input_units, output_units)))
                self.biases.append(np.random.normal(0, std_dev, output_units))
            # Initialize weights and biases with He initialization
            elif self.initialization == 'he':
                std_dev = np.sqrt(2 / input_units)
                self.weights.append(np.random.normal(0, std_dev, (input_units, output_units)))
                self.biases.append(np.random.normal(0, std_dev, output_units))

            else:  # Default initialization is  random
                self.weights.append(np.random.randn(input_units, output_units))
                self.biases.append(np.random.randn(output_units))

    # This function performs forward propagation in the MLP.
    # It calculates the activations of each layer and stores them.
    def _forward(self, X):
        activations = [X]
        inputs = []

        for W, b in zip(self.weights, self.biases):
            Z = np.dot(activations[-1], W) + b
            inputs.append(Z)
            if self.activation == 'sigmoid':
                activations.append(_sigmoid(Z))
            elif self.activation == 'relu':
                activations.append(_relu(Z))

        return activations, inputs

    # This function performs backward propagation in the MLP.
    # It computes the gradients of the weights and biases in the MLP.
    def _backward(self, X, y, activations, inputs):
        m = X.shape[0]

        if self.activation == 'sigmoid' and self.loss_func == "rmse":
            dZ = _rmse_derivative(y, activations[-1]) * _sigmoid_derivative(inputs[-1])
        elif self.activation == 'relu' and self.loss_func == "rmse":
            dZ = _rmse_derivative(y, activations[-1]) * _relu_derivative(inputs[-1])
        elif self.activation == 'relu' and self.loss_func == "mse":
            dZ = _mse_derivative(y, activations[-1]) * _relu_derivative(inputs[-1])
        else:  # default: sigmoid and mse
            dZ = _mse_derivative(y, activations[-1]) * _sigmoid_derivative(inputs[-1])
        dW = 1 / m * np.dot(activations[-2].T, dZ)
        db = 1 / m * np.sum(dZ, axis=0)

        gradients = [(dW, db)]

        for i in range(len(self.hidden_layer_sizes) - 1, -1, -1):
            if self.activation == 'sigmoid':
                dZ = np.dot(dZ, self.weights[i + 1].T) * _sigmoid_derivative(inputs[i])
            elif self.activation == 'relu':
                dZ = np.dot(dZ, self.weights[i + 1].T) * _relu_derivative(inputs[i])
            dW = 1 / m * np.dot(activations[i].T, dZ)
            db = 1 / m * np.sum(dZ, axis=0)

            gradients.append((dW, db))

        return gradients[::-1]

    # Update the parameters using the Adam optimizer
    def _update_parameters(self, gradients, t, m_t_weights, m_t_biases, v_t_weights, v_t_biases):
        for i in range(len(self.weights)):
            # Update the first and second moments for the weights and biases
            m_t_weights[i] = self.beta1 * m_t_weights[i] + (1 - self.beta1) * gradients[i][0]
            v_t_weights[i] = self.beta2 * v_t_weights[i] + (1 - self.beta2) * (gradients[i][0] ** 2)
            m_t_biases[i] = self.beta1 * m_t_biases[i] + (1 - self.beta1) * gradients[i][1]
            v_t_biases[i] = self.beta2 * v_t_biases[i] + (1 - self.beta2) * (gradients[i][1] ** 2)

            # Calculate bias-corrected first and second moments for weights and biases
            m_hat_weights = m_t_weights[i] / (1 - self.beta1 ** t)
            v_hat_weights = v_t_weights[i] / (1 - self.beta2 ** t)
            m_hat_biases = m_t_biases[i] / (1 - self.beta1 ** t)
            v_hat_biases = v_t_biases[i] / (1 - self.beta2 ** t)

            # Update weights and biases using the Adam optimizer
            self.weights[i] -= self.learning_rate * m_hat_weights / (np.sqrt(v_hat_weights) + self.epsilon)
            self.biases[i] -= self.learning_rate * m_hat_biases / (np.sqrt(v_hat_biases) + self.epsilon)

    # Fit method for training
    def fit(self, X, y, val_split=0.2, sample_size=20):
        # Setting layer sizes, including input, hidden layers, and output layer
        layer_sizes = [X.shape[1]] + list(self.hidden_layer_sizes) + [1]  # output size is 1 as predicted value

        # Initializing the weights and biases
        self._initialize_parameters(layer_sizes)

        # Split the training set into training and validation sets
        split_index = int((1 - val_split) * X.shape[0])
        X_val = X[split_index:]
        y_val = y[split_index:]
        new_X_train = X[:split_index]
        new_y_train = y[:split_index]

        # Initializing Adam optimizer variables
        m_weights = [np.zeros_like(w) for w in self.weights]
        v_weights = [np.zeros_like(w) for w in self.weights]
        m_biases = [np.zeros_like(b) for b in self.biases]
        v_biases = [np.zeros_like(b) for b in self.biases]

        start_fitting_time = time.time()  # Start measuring time
        for epoch in range(self.max_iter):
            # Perform forward propagation and store activations and inputs
            activations, inputs = self._forward(new_X_train)

            # Perform backward propagation and compute gradients
            gradients = self._backward(new_X_train, new_y_train, activations, inputs)

            # Update parameters using the Adam optimizer
            self._update_parameters(gradients, epoch + 1, m_weights, m_biases, v_weights, v_biases)

            # Appending the training and validation MSE and R2 score values to the lists
            # Calculate the loss value
            if self.loss_func == "rmse":
                loss_value = _rmse(new_y_train, self.predict(new_X_train))
            else:  # Default: mse
                loss_value = _mse(new_y_train, self.predict(new_X_train))
            self.loss_values.append(loss_value)

            # Calculating R2 score for the validation set
            val_r2_score = r_squared_score(y_val, self.predict(X_val))
            self.val_scores.append(val_r2_score)

            # Computing running averages
            if epoch >= sample_size:
                # Taking mean of the sampled values
                running_avg_loss = np.mean(self.loss_values[epoch - sample_size:epoch])
                running_avg_val_score = np.mean(self.val_scores[epoch - sample_size:epoch])
                # Adding the sampled values to the lists
                self.running_avg_loss_values.append(running_avg_loss)
                self.running_avg_val_scores.append(running_avg_val_score)

            # Print the training and validation performance at each epoch
            if epoch % 10 == 0:
                if self.loss_func == "rmse":
                    print("-------------------------------------------------")
                    train_rmse = loss_value
                    print(f'Epoch {epoch}: Training loss: {train_rmse}')
                else:  # default loss function is mse
                    print("-------------------------------------------------")
                    train_mse = loss_value
                    print(f'Epoch {epoch}: Training loss: {train_mse}')

                # Calculate R2 score for the validation set
                print(f'Epoch {epoch}: Validation Score: {val_r2_score}')
                print("-------------------------------------------------")

        end_fitting_time = time.time()  # End measuring time
        fitting_time = end_fitting_time - start_fitting_time
        print(f"Fitting time: {fitting_time:.4f} seconds")

    # Perform forward propagation and return the final layer's activation
    def predict(self, X):
        activations, _ = self._forward(X)
        return activations[-1]

    # Method for plotting running average loss values and validation scores
    def plot_scores_and_losses(self, sample_size=20):

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(range(sample_size, len(self.val_scores)), self.running_avg_val_scores)
        plt.xlabel("Iteration")
        plt.ylabel("Running Average Validation R-squared score")
        plt.title("Running Average Validation R-squared score vs. iterations")

        plt.subplot(1, 2, 2)
        plt.plot(range(sample_size, len(self.loss_values)), self.running_avg_loss_values)
        plt.xlabel("Iteration")
        plt.ylabel("Running Average Loss value")
        plt.title("Running Average Loss value vs. iterations")

        plt.show()


# Example usage
if __name__ == "__main__":
    # Importing dataset
    kc_dataset = pd.read_csv(r'./Data/kc_house_data.csv')

    # Splitting train test data
    X_train, X_test, y_train, y_test = data_preprocess.preprocess(kc_dataset, outlier_removal=True)

    # Reshaping y_train and y_test
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    # Creating a multi layer perceptron regressor with
    # specified hidden layer sizes, learning rate, max iterations, activation, initialization and loss function
    mlp_regressor = MLPRegressor(hidden_layer_sizes=(40, 20, 10, 2), learning_rate=0.03, max_iter=2000,
                                 activation="sigmoid", initialization="he", loss_func="rmse")

    # Train the regressor on the dataset
    mlp_regressor.fit(X_train, y_train)

    # Predicting and testing the model
    predictions = mlp_regressor.predict(X_test)
    testing_model.test(y_test, predictions, "my mlp results")
    mlp_regressor.plot_scores_and_losses()
