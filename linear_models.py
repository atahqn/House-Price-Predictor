import numpy as np
import matplotlib.pyplot as plt
import time

import pandas as pd

import data_preprocess
import testing_model


# Defining the loss function as mean squared error
def loss_func(y_train_prediction, y_train):
    mse = np.mean((y_train_prediction - y_train) ** 2)
    return mse


# Defining R-squared score function
def r_squared_score(y_true, y_prediction):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_prediction) ** 2)
    return 1 - (ss_res / ss_total)


# Defining Linear Regression class
class LinearRegression:
    def __init__(self, lr=0.5, n_iters=1000, penalty=None, validation_split=0.2):
        self.lr = lr
        self.n_iters = n_iters
        self.penalty = penalty
        self.weights = None
        self.bias = None
        self.validation_split = validation_split
        self.val_scores = []
        self.loss_values = []

    # Fit method for training the model
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Splitting the dataset into training and validation sets
        train_indices = np.random.choice(n_samples, int(n_samples * (1 - self.validation_split)), replace=False)
        val_indices = np.array(list(set(range(n_samples)) - set(train_indices)))

        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]

        start_fitting_time = time.time()  # Start measuring time

        # Performing gradient descent algorithm to update weights and bias
        for _ in range(self.n_iters):
            y_train_prediction = np.dot(X_train, self.weights) + self.bias

            dw = (1 / n_samples) * np.dot(X_train.T, (y_train_prediction - y_train))
            db = (1 / n_samples) * np.sum(y_train_prediction - y_train)

            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

            # Evaluate the model on the validation set
            y_val_prediction = self.predict(X_val)
            val_score = r_squared_score(y_val, y_val_prediction)
            self.val_scores.append(val_score)

            # Calculate the loss value
            loss_value = loss_func(y_train_prediction, y_train)
            self.loss_values.append(loss_value)
        end_fitting_time = time.time()  # End measuring time
        fitting_time = end_fitting_time - start_fitting_time
        print(f"Fitting time: {fitting_time:.4f} seconds")

    # Prediction method for the model
    def predict(self, X):
        y_prediction = np.dot(X, self.weights) + self.bias
        return y_prediction

    # Method to plot validation scores and loss values during training
    def plot_scores_and_losses(self):

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(self.val_scores)
        plt.xlabel("Iteration")
        plt.ylabel("Validation R-squared score")
        plt.title("Validation R-squared score vs. iterations")

        plt.subplot(1, 2, 2)
        plt.plot(self.loss_values)
        plt.xlabel("Iteration")
        plt.ylabel("Loss value")
        plt.title("Loss value vs. iterations")

        plt.show()


if __name__ == "__main__":
    # Importing dataset
    kc_dataset = pd.read_csv(r'./Data/kc_house_data.csv')

    # Splitting train test data
    X_train, X_test, y_train, y_test = data_preprocess.preprocess(kc_dataset, outlier_removal=True)

    # Fitting data to my linear regression model
    MyLinearRegression = LinearRegression()
    MyLinearRegression.fit(X_train, y_train)
    y_prediction = MyLinearRegression.predict(X_test)
    testing_model.test(y_test, y_prediction, "My Linear Regression")
    MyLinearRegression.plot_scores_and_losses()
