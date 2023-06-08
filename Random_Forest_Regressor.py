import time
import numpy as np
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import pandas as pd

from linear_models import r_squared_score
import data_preprocess
import testing_model

np.random.seed(6)


# Defining Node class for building the decision tree
class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, var_red=None, value=None):
        # for decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.var_red = var_red  # variance reduction

        # for leaf node
        self.value = value

    # To make system multithreading and parallel we need to add following methods to make it serialize and deserialized
    # Defining the state of the object for serialization purposes
    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    # Defining how the object is deserialized
    def __setstate__(self, state):
        self.__dict__.update(state)


# Defining the DecisionTreeRegressor class
class DecisionTreeRegressor:
    def __init__(self, min_samples_split=2, max_depth=2):
        # Initializing the root of the tree and stopping/ splitting conditions
        self.root = None
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    # Function to build the decision tree
    def build_tree(self, dataset, curr_depth=0):
        # Split the dataset into features and labels
        X, y = dataset[:, :-1], dataset[:, -1]
        num_samples, num_features = np.shape(X)
        best_split = {"var_red": -float("inf")}  # To avoid an error first key must be very small number
        max_var_red = -float("inf")  # Also giving the minimum number for max variance reduction at first

        # Check stopping conditions before splitting
        if num_samples >= self.min_samples_split and curr_depth <= self.max_depth:
            # Find the best split for the current node
            best_split = self.get_best_split(dataset, num_samples, num_features)

            # Check if the variance reduction is positive
            if best_split["var_red"] > 0:
                # Recursively build the left subtree
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth + 1)

                # Recursively build the right subtree
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth + 1)

                # Return the decision node
                return Node(best_split["feature_index"], best_split["threshold"],
                            left_subtree, right_subtree, best_split["var_red"])

        # Compute the leaf node value
        leaf_value = self.calculate_leaf_value(y)

        # Return the leaf node
        return Node(value=leaf_value)

    # Function to find the best split for a given dataset
    def get_best_split(self, dataset, num_samples, num_features):
        best_split = {"var_red": -float("inf")}  # To avoid an error first key must be very small number
        max_var_red = -float("inf")  # Also giving the minimum number for max variance reduction at first

        # Loop through all features and find the best split for each feature
        for feature_index in range(num_features):
            result = self.find_best_split_for_feature(dataset, feature_index, num_samples)

            # Update the best split if a better one is found
            if result["var_red"] > max_var_red:
                best_split = result
                max_var_red = result["var_red"]

        return best_split

    # Function to find the best split for a given feature
    def find_best_split_for_feature(self, dataset, feature_index, num_samples):
        best_split = {"var_red": -float("inf")}  # To avoid an error first key must be very small number
        max_var_red = -float("inf")  # Also giving the minimum number for max variance reduction at first

        # Get unique values for the current feature
        feature_values = dataset[:, feature_index]
        possible_thresholds = np.unique(feature_values)

        # Calculate the parent variance
        parent_var = np.var(dataset[:, -1])

        # Loop through all possible thresholds for the current feature
        for threshold in possible_thresholds:
            # Create a mask for splitting the dataset
            mask = dataset[:, feature_index] <= threshold
            dataset_left, dataset_right = dataset[mask], dataset[~mask]

            # Check if both left and right datasets are non-empty
            if len(dataset_left) > 0 and len(dataset_right) > 0:
                # Calculate the variances for left and right datasets
                left_var, right_var = np.var(dataset_left[:, -1]), np.var(dataset_right[:, -1])
                left_weight, right_weight = len(dataset_left) / num_samples, len(dataset_right) / num_samples

                # Calculate the current variance reduction
                curr_var_red = parent_var - (left_weight * left_var + right_weight * right_var)

                # Update the best split if a better one is found
                if curr_var_red > max_var_red:
                    best_split["feature_index"] = feature_index
                    best_split["threshold"] = threshold
                    best_split["dataset_left"] = dataset_left
                    best_split["dataset_right"] = dataset_right
                    best_split["var_red"] = curr_var_red
                    max_var_red = curr_var_red

        return best_split

    # Function to split the dataset based on the feature index and threshold
    def split(self, dataset, feature_index, threshold):
        dataset_left = np.array([row for row in dataset if row[feature_index] <= threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index] > threshold])
        return dataset_left, dataset_right

    # Function to compute variance reduction
    def variance_reduction(self, parent, l_child, r_child):
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        reduction = np.var(parent) - (weight_l * np.var(l_child) + weight_r * np.var(r_child))
        return reduction

    # Function to compute the leaf node value
    def calculate_leaf_value(self, Y):
        val = np.mean(Y)
        return val

    # Function to fit the decision tree to the training data
    def fit(self, X, Y):
        dataset = np.concatenate((X, Y.reshape(-1, 1)), axis=1)
        self.root = self.build_tree(dataset)
        print("Tree is built (data is fitted)")

    # Function to make a prediction for a single data point
    def make_prediction(self, x, tree):
        if tree.value is not None: return tree.value
        feature_val = x[tree.feature_index]
        if feature_val <= tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)

    # Function to make predictions for an array of data points
    def predict(self, X):
        predictions = [self.make_prediction(x, self.root) for x in X]
        return predictions


# Define the RandomForestRegressor class
class RandomForestRegressor:
    def __init__(self, n_estimators=100, min_samples_split=2, max_depth=2, validation_split=0.2):
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.validation_split = validation_split
        self.trees = []
        self.val_scores = []
        self.feature_importances_ = None

    # Function to fit a single decision tree to the dataset
    def fit_tree(self, X, y):
        indices = np.random.randint(0, len(X), len(X))
        X_sample, y_sample = X[indices], y[indices]

        tree = DecisionTreeRegressor(min_samples_split=self.min_samples_split,
                                     max_depth=self.max_depth)
        tree.fit(X_sample, y_sample)
        return tree

    # This function fits the random forest model to the dataset.
    def fit(self, X, y):
        # Initialize lists to store the trees and their validation scores
        self.trees = []
        self.val_scores = []

        # Split the dataset into training and validation sets by randomly selecting indices for each set
        train_indices = np.random.choice(len(X), int(len(X) * (1 - self.validation_split)), replace=False)
        val_indices = np.array(list(set(range(len(X))) - set(train_indices)))

        # Create training and validation sets using the indices
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]

        # Record the starting time
        start_time = time.time()

        # Fit the specified number of trees in parallel using job lib's Parallel and delayed functions
        self.trees = Parallel(n_jobs=-1)(
            delayed(self.fit_tree)(X_train, y_train) for _ in range(self.n_estimators)
        )

        # Record the ending time
        end_time = time.time()

        # Calculate the time taken to fit the trees
        elapsed_time = end_time - start_time

        # Evaluate the model on the validation set by calculating the R-squared score for each tree
        for tree in self.trees:
            val_predictions = tree.predict(X_val)
            val_score = r_squared_score(y_val, val_predictions)
            self.val_scores.append(val_score)

        # Print the average R-squared score and time taken to fit the trees
        print("Average validation R-squared score: ", np.mean(self.val_scores) * 100, "%")
        print("Time elapsed during fitting: {:.2f} seconds".format(elapsed_time))

        # Compute the feature importances after fitting all trees
        self.feature_importances_ = self.compute_feature_importance(X.shape[1])

    # This function computes the feature importances by averaging the importances across all trees
    def compute_feature_importance(self, num_features):
        feature_importance = np.zeros(num_features)
        for tree in self.trees:
            importance = self.calculate_tree_feature_importance(tree, num_features)
            feature_importance += importance
        return feature_importance / self.n_estimators

    # This function calculates the feature importances for a single tree by traversing its nodes
    def calculate_tree_feature_importance(self, tree, num_features):
        importance = np.zeros(num_features)
        nodes = [tree.root]
        while nodes:
            current = nodes.pop()
            # If the current node has a feature split, add its contribution to the feature importance
            if current.var_red is not None:
                importance[current.feature_index] += current.var_red
            # Traversing left and right child nodes if they exist
            if current.left is not None:
                nodes.append(current.left)
            if current.right is not None:
                nodes.append(current.right)
        return importance

    # Function to make predictions with the random forest
    def predict(self, X):
        predictions = np.zeros(len(X))

        # Use job lib's Parallel and delayed functions for parallel predictions
        tree_predictions = Parallel(n_jobs=-1)(
            delayed(tree.predict)(X) for tree in self.trees
        )

        for prediction in tree_predictions:
            predictions += prediction

        return predictions / self.n_estimators

    # Function to plot the validation scores
    def plot_validation_scores(self):
        plt.figure(figsize=(12, 6))

        # Plot individual validation scores
        plt.scatter(range(len(self.val_scores)), self.val_scores, label='Individual scores', alpha=0.5)

        # Calculate and plot cumulative moving average
        cumulative_avg_scores = np.cumsum(self.val_scores) / (np.arange(len(self.val_scores)) + 1)
        plt.plot(cumulative_avg_scores, color='red', label='Cumulative average')

        plt.xlabel("Estimator")
        plt.ylabel("Validation R-squared score")
        plt.title("Validation R-squared score vs. Estimators")
        plt.legend()
        plt.grid()
        plt.show()


def feature_main():
    # Importing dataset
    kc_dataset = pd.read_csv(r'./Data/kc_house_data.csv')

    # Splitting train test data
    X_train, X_test, y_train, y_test = data_preprocess.preprocess(kc_dataset, outlier_removal=True)

    # Fitting data to my Random Forest Regressor model
    my_rfr = RandomForestRegressor(n_estimators=3, min_samples_split=3, max_depth=12)
    my_rfr.fit(X_train, y_train)

    my_rfr_predictions = my_rfr.predict(X_test)
    # Evaluate the first model
    testing_model.test(y_test, my_rfr_predictions, "My Random Forest first predictions")

    # Calculate feature importances
    feature_importances = my_rfr.feature_importances_

    # Plot the normalized feature importance's for the first model
    normalized_importances = feature_importances / np.sum(feature_importances)
    plt.figure(figsize=(12, 6))
    plt.bar(range(X_test.shape[1]), normalized_importances[np.argsort(feature_importances)[::-1]], align='center')
    plt.xticks(range(X_test.shape[1]), np.argsort(feature_importances)[::-1])
    plt.xlabel('Feature Index')
    plt.ylabel('Normalized Importance')
    plt.title('Feature Importances for First Model')
    plt.show()

    # Setting the number of top features to keep
    num_top_features = 15

    # Getting the indices of the top features
    top_feature_indices = np.argsort(feature_importances)[::-1][:num_top_features]

    # Selecting the top features from the dataset
    X_train_top_features = X_train[:, top_feature_indices]
    X_test_top_features = X_test[:, top_feature_indices]

    # Fitting the model using only the top features
    my_rfr_top_features = RandomForestRegressor(n_estimators=3, min_samples_split=3, max_depth=12)
    my_rfr_top_features.fit(X_train_top_features, y_train)

    # Predicting using the updated model
    my_rfr_prediction_top_features = my_rfr_top_features.predict(X_test_top_features)

    # Evaluating the updated model
    testing_model.test(y_test, my_rfr_prediction_top_features, "My Random Forest with Top Features")

    # Plotting the normalized feature importance's for the second model
    feature_importances_second_model = my_rfr_top_features.feature_importances_
    normalized_importances_second_model = feature_importances_second_model / np.sum(feature_importances_second_model)
    plt.figure(figsize=(12, 6))
    plt.bar(range(X_test_top_features.shape[1]),
            normalized_importances_second_model[np.argsort(feature_importances_second_model)[::-1]], align='center')
    plt.xticks(range(X_test_top_features.shape[1]), np.argsort(feature_importances_second_model)[::-1])
    plt.xlabel('Feature Index')
    plt.ylabel('Normalized Importance')
    plt.title('Feature Importances for Second Model')
    plt.show()


def first_main():
    # Importing dataset
    kc_dataset = pd.read_csv(r'./Data/kc_house_data.csv')

    # Splitting train test data
    X_train, X_test, y_train, y_test = data_preprocess.preprocess(kc_dataset, outlier_removal=True)

    # Fitting data to my Random Forest Regressor model
    my_rfr = RandomForestRegressor(n_estimators=3, min_samples_split=3, max_depth=12)
    my_rfr.fit(X_train, y_train)

    my_rfr_predictions = my_rfr.predict(X_test)
    # Evaluate the first model
    testing_model.test(y_test, my_rfr_predictions, "My Random Forest first predictions")


if __name__ == "__main__":
    feature_main()


