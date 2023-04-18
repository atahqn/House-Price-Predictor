import numpy as np
from joblib import Parallel, delayed


class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, var_red=None, value=None):
        # for decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.var_red = var_red

        # for leaf node
        self.value = value

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


class DecisionTreeRegressor:
    def __init__(self, min_samples_split=2, max_depth=2):

        # initialize the root of the tree
        self.root = None

        # stopping conditions
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    def build_tree(self, dataset, curr_depth=0):

        X, Y = dataset[:, :-1], dataset[:, -1]
        num_samples, num_features = np.shape(X)
        best_split = {"var_red": -float("inf")}
        max_var_red = -float("inf")
        # split until stopping conditions are met
        if num_samples >= self.min_samples_split and curr_depth <= self.max_depth:
            # find the best split
            best_split = self.get_best_split(dataset, num_samples, num_features)
            # check if information gain is positive
            if best_split["var_red"] > 0:
                # recur left
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth + 1)
                # recur right
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth + 1)
                # return decision node
                return Node(best_split["feature_index"], best_split["threshold"],
                            left_subtree, right_subtree, best_split["var_red"])

        # compute leaf node
        leaf_value = self.calculate_leaf_value(Y)
        # return leaf node
        return Node(value=leaf_value)

    def get_best_split(self, dataset, num_samples, num_features):
        best_split = {"var_red": -float("inf")}
        max_var_red = -float("inf")

        for feature_index in range(num_features):
            result = self.find_best_split_for_feature(dataset, feature_index, num_samples)
            if result["var_red"] > max_var_red:
                best_split = result
                max_var_red = result["var_red"]

        return best_split

    def find_best_split_for_feature(self, dataset, feature_index, num_samples):
        best_split = {"var_red": -float("inf")}
        max_var_red = -float("inf")

        feature_values = dataset[:, feature_index]
        possible_thresholds = np.unique(feature_values)

        parent_var = np.var(dataset[:, -1])

        for threshold in possible_thresholds:
            mask = dataset[:, feature_index] <= threshold
            dataset_left, dataset_right = dataset[mask], dataset[~mask]

            if len(dataset_left) > 0 and len(dataset_right) > 0:
                left_var, right_var = np.var(dataset_left[:, -1]), np.var(dataset_right[:, -1])
                left_weight, right_weight = len(dataset_left) / num_samples, len(dataset_right) / num_samples

                curr_var_red = parent_var - (left_weight * left_var + right_weight * right_var)

                if curr_var_red > max_var_red:
                    best_split["feature_index"] = feature_index
                    best_split["threshold"] = threshold
                    best_split["dataset_left"] = dataset_left
                    best_split["dataset_right"] = dataset_right
                    best_split["var_red"] = curr_var_red
                    max_var_red = curr_var_red

        return best_split

    def split(self, dataset, feature_index, threshold):
        """ function to split the data """

        dataset_left = np.array([row for row in dataset if row[feature_index] <= threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index] > threshold])
        return dataset_left, dataset_right

    def variance_reduction(self, parent, l_child, r_child):
        """ function to compute variance reduction """

        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        reduction = np.var(parent) - (weight_l * np.var(l_child) + weight_r * np.var(r_child))
        return reduction

    def calculate_leaf_value(self, Y):
        ''' function to compute leaf node '''

        val = np.mean(Y)
        return val

    def fit(self, X, Y):
        """ function to train the tree """
        # print("shape of X is ",X.shape)
        # print("shape of y is ", Y.shape)
        dataset = np.concatenate((X, Y.reshape(-1, 1)), axis=1)
        self.root = self.build_tree(dataset)
        print("Tree is built (data is fitted)")

    def make_prediction(self, x, tree):
        """ function to predict new dataset """

        if tree.value is not None: return tree.value
        feature_val = x[tree.feature_index]
        if feature_val <= tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)

    def predict(self, X):
        """ function to predict a single data point """
        predictions = [self.make_prediction(x, self.root) for x in X]
        return predictions


class RandomForestRegressor:
    def __init__(self, n_estimators=100, min_samples_split=2, max_depth=2):
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.trees = []

    def fit_tree(self, X, y):
        indices = np.random.randint(0, len(X), len(X))
        X_sample, y_sample = X[indices], y[indices]

        tree = DecisionTreeRegressor(min_samples_split=self.min_samples_split,
                                     max_depth=self.max_depth)
        tree.fit(X_sample, y_sample)
        return tree

    def fit(self, X, y):
        self.trees = []

        # Use job lib's Parallel and delayed functions to fit trees in parallel
        self.trees = Parallel(n_jobs=-1)(
            delayed(self.fit_tree)(X, y) for _ in range(self.n_estimators)
        )

    def predict(self, X):
        predictions = np.zeros(len(X))

        # Use job lib's Parallel and delayed functions for parallel predictions
        tree_predictions = Parallel(n_jobs=-1)(
            delayed(tree.predict)(X) for tree in self.trees
        )

        for pred in tree_predictions:
            predictions += pred

        return predictions / self.n_estimators
