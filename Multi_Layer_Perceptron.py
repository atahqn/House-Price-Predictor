import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import time

from linear_models import r_squared_score
import data_preprocess
import testing_model


# np.random.seed(1)


class MLPRegressor:
    def __init__(self, hidden_layer_sizes=(100,), learning_rate=0.01, max_iter=1000, random_state=None, beta1=0.9,
                 beta2=0.999, epsilon=1e-8, activation="relu", initialization="random", loss_func="mse"):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state
        self.weights = []
        self.biases = []
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.activation = activation
        self.loss_func = loss_func
        self.initialization = initialization
        self.val_scores = []
        self.loss_values = []

    def _initialize_parameters(self, layer_sizes):
        if self.random_state:
            np.random.seed(self.random_state)

        for i in range(len(layer_sizes) - 1):
            input_units = layer_sizes[i]
            output_units = layer_sizes[i + 1]

            if self.initialization == 'zeros':
                self.weights.append(np.zeros((input_units, output_units)))
                self.biases.append(np.zeros(output_units))

            elif self.initialization == 'ones':
                self.weights.append(np.ones((input_units, output_units)))
                self.biases.append(np.ones(output_units))

            elif self.initialization == 'xavier':
                std_dev = np.sqrt(1 / input_units)
                self.weights.append(np.random.normal(0, std_dev, (input_units, output_units)))
                self.biases.append(np.random.normal(0, std_dev, output_units))

            elif self.initialization == 'he':
                std_dev = np.sqrt(2 / input_units)
                self.weights.append(np.random.normal(0, std_dev, (input_units, output_units)))
                self.biases.append(np.random.normal(0, std_dev, output_units))

            else:  # Default: random
                self.weights.append(np.random.randn(input_units, output_units))
                self.biases.append(np.random.randn(output_units))

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoid_derivative(self, x):
        return self._sigmoid(x) * (1 - self._sigmoid(x))

    def _mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def _mse_derivative(self, y_true, y_pred):
        return -2 * (y_true - y_pred)

    def _rmse(self, y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred) ** 2))

    def _rmse_derivative(self, y_true, y_pred):
        return -1 * (y_true - y_pred) / self._rmse(y_true, y_pred)

    def _relu(self, x):
        return np.maximum(0, x)

    def _relu_derivative(self, x):
        return (x > 0).astype(float)

    def _forward(self, X):
        activations = [X]
        inputs = []

        for W, b in zip(self.weights, self.biases):
            Z = np.dot(activations[-1], W) + b
            inputs.append(Z)
            if self.activation == 'sigmoid':
                activations.append(self._sigmoid(Z))
            elif self.activation == 'relu':
                activations.append(self._relu(Z))

        return activations, inputs

    def _backward(self, X, y, activations, inputs):
        m = X.shape[0]

        if self.activation == 'sigmoid' and self.loss_func == "rmse":
            dZ = self._rmse_derivative(y, activations[-1]) * self._sigmoid_derivative(inputs[-1])
        elif self.activation == 'relu' and self.loss_func == "rmse":
            dZ = self._rmse_derivative(y, activations[-1]) * self._relu_derivative(inputs[-1])
        elif self.activation == 'relu' and self.loss_func == "mse":
            dZ = self._mse_derivative(y, activations[-1]) * self._relu_derivative(inputs[-1])
        else:  # default: sigmoid and mse
            dZ = self._mse_derivative(y, activations[-1]) * self._sigmoid_derivative(inputs[-1])
        dW = 1 / m * np.dot(activations[-2].T, dZ)
        db = 1 / m * np.sum(dZ, axis=0)

        gradients = [(dW, db)]

        for i in range(len(self.hidden_layer_sizes) - 1, -1, -1):
            if self.activation == 'sigmoid':
                dZ = np.dot(dZ, self.weights[i + 1].T) * self._sigmoid_derivative(inputs[i])
            elif self.activation == 'relu':
                dZ = np.dot(dZ, self.weights[i + 1].T) * self._relu_derivative(inputs[i])
            dW = 1 / m * np.dot(activations[i].T, dZ)
            db = 1 / m * np.sum(dZ, axis=0)

            gradients.append((dW, db))

        return gradients[::-1]

    def _update_parameters(self, gradients, t, m_t, v_t):
        for i in range(len(self.weights)):
            m_t[i] = self.beta1 * m_t[i] + (1 - self.beta1) * gradients[i][0]
            v_t[i] = self.beta2 * v_t[i] + (1 - self.beta2) * (gradients[i][0] ** 2)

            m_hat = m_t[i] / (1 - self.beta1 ** t)
            v_hat = v_t[i] / (1 - self.beta2 ** t)

            self.weights[i] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            self.biases[i] -= self.learning_rate * gradients[i][1]

    def fit(self, X, y, val_split=0.2):
        layer_sizes = [X.shape[1]] + list(self.hidden_layer_sizes) + [1]  # output size is 1 as predicted value
        self._initialize_parameters(layer_sizes)

        # Split the training set into training and validation sets
        split_index = int((1 - val_split) * X.shape[0])
        X_val = X[split_index:]
        y_val = y[split_index:]
        X_train = X[:split_index]
        y_train = y[:split_index]

        # Initialize Adam optimizer variables
        m_weights = [np.zeros_like(w) for w in self.weights]
        v_weights = [np.zeros_like(w) for w in self.weights]
        m_biases = [np.zeros_like(b) for b in self.biases]
        v_biases = [np.zeros_like(b) for b in self.biases]

        start_fitting_time = time.time()  # Start measuring time
        for epoch in range(self.max_iter):
            activations, inputs = self._forward(X_train)
            gradients = self._backward(X_train, y_train, activations, inputs)

            # Update parameters with Adam optimizer
            for i, (dw, db) in enumerate(gradients):
                m_weights[i] = self.beta1 * m_weights[i] + (1 - self.beta1) * dw
                v_weights[i] = self.beta2 * v_weights[i] + (1 - self.beta2) * (dw ** 2)
                m_biases[i] = self.beta1 * m_biases[i] + (1 - self.beta1) * db
                v_biases[i] = self.beta2 * v_biases[i] + (1 - self.beta2) * (db ** 2)

                m_weights_corr = m_weights[i] / (1 - self.beta1 ** (epoch + 1))
                v_weights_corr = v_weights[i] / (1 - self.beta2 ** (epoch + 1))
                m_biases_corr = m_biases[i] / (1 - self.beta1 ** (epoch + 1))
                v_biases_corr = v_biases[i] / (1 - self.beta2 ** (epoch + 1))

                self.weights[i] -= self.learning_rate * m_weights_corr / (np.sqrt(v_weights_corr) + self.epsilon)
                self.biases[i] -= self.learning_rate * m_biases_corr / (np.sqrt(v_biases_corr) + self.epsilon)
            # Append the training and validation MSE and R2 score values to the lists
            # Calculate the loss value
            if self.loss_func == "rmse":
                loss_value = self._rmse(y_train, self.predict(X_train))
            else:  # Default: mse
                loss_value = self._mse(y_train, self.predict(X_train))
            self.loss_values.append(loss_value)

            # Calculate R2 score for the validation set
            val_r2_score = r_squared_score(y_val, self.predict(X_val))
            self.val_scores.append(val_r2_score)

            # Print the training and validation performance at each epoch
            if epoch % 10 == 0:
                if self.loss_func == "rmse":
                    print("-------------------------------------------------")
                    train_rmse = self._rmse(y_train, self.predict(X_train))
                    print(f'Epoch {epoch}: Training loss: {train_rmse}')
                else:
                    print("-------------------------------------------------")
                    train_mse = self._mse(y_train, self.predict(X_train))
                    print(f'Epoch {epoch}: Training loss: {train_mse}')

                # Calculate R2 score for the validation set
                val_r2_score = r_squared_score(y_val, self.predict(X_val))
                print(f'Epoch {epoch}: Validation Score: {val_r2_score}')
                print("-------------------------------------------------")
        end_fitting_time = time.time()  # End measuring time
        fitting_time = end_fitting_time - start_fitting_time
        print(f"Fitting time: {fitting_time:.4f} seconds")

    def predict(self, X):
        activations, _ = self._forward(X)
        return activations[-1]

    def plot_scores_and_losses(self):

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(range(20, len(self.val_scores)), self.val_scores[20:])
        plt.xlabel("Iteration")
        plt.ylabel("Validation R-squared score")
        plt.title("Validation R-squared score vs. iterations")
        # plt.ylim(-1, 1)  # Set custom y-axis limits here

        plt.subplot(1, 2, 2)
        plt.plot(range(20, len(self.loss_values)), self.loss_values[20:])
        plt.xlabel("Iteration")
        plt.ylabel("Loss value")
        plt.title("Loss value vs. iterations")

        plt.show()


# Example usage
if __name__ == "__main__":
    # Importing dataset
    kc_dataset = pd.read_csv(r'./Data/kc_house_data.csv')

    # Splitting train test data
    X_train, X_test, y_train, y_test = data_preprocess.preprocess(kc_dataset)

    # Reshape y_train and y_test
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    # Create a regressor with specified layer sizes, learning rate, and epochs
    # Create a regressor with specified hidden layer sizes, learning rate, and max iterations
    mlp_regressor = MLPRegressor(hidden_layer_sizes=(16, 8), learning_rate=0.05, max_iter=1000, activation="sigmoid",
                                 initialization="random", loss_func="mse")

    # Train the regressor on the dataset
    mlp_regressor.fit(X_train, y_train)

    # Predict on new data
    predictions = mlp_regressor.predict(X_test)
    testing_model.test(y_test, predictions, "my mlp results")
    mlp_regressor.plot_scores_and_losses()

