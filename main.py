import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import data_preprocess
import Random_Forest_Regressor
import models
import testing_model

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import (Lasso, Ridge)


def sklearnModelsResults(X_train, X_test, y_train, y_test):
    slr = LinearRegression()
    slr.fit(X_train, y_train)
    slr_prediction = slr.predict(X_test)
    testing_model.test(y_test, slr_prediction, "Built-in Linear Regression")

    rg = Ridge(alpha=0.1)
    rg.fit(X_train, y_train)
    rg_prediction = rg.predict(X_test)
    testing_model.test(y_test, rg_prediction, "Built-in Ridge Regression")

    ls = Lasso(alpha=0.0001)
    ls.fit(X_train, y_train)
    ls_prediction = ls.predict(X_test)
    testing_model.test(y_test, ls_prediction, "Built-in Lasso Regression")

    sk_rfr = RandomForestRegressor(n_jobs=-1, n_estimators=3, min_samples_split=2, max_depth=6)
    sk_rfr.fit(X_train, y_train)
    sk_rfr_prediction = sk_rfr.predict(X_test)
    testing_model.test(y_test, sk_rfr_prediction, "Built-in Random Forest")


def myModelsResults(X_train, X_test, y_train, y_test):

    # Fitting data to my linear regression model
    CustomLinearRegression = models.LinearRegression()
    CustomLinearRegression.fit(X_train, y_train)
    y_prediction = CustomLinearRegression.predict(X_test)
    testing_model.test(y_test, y_prediction, "My Linear Regression")
    # CustomLinearRegression.plot_scores_and_losses()

    # Fitting data to my Random Forest Regressor model
    my_rfr = Random_Forest_Regressor.RandomForestRegressor(n_estimators=3, min_samples_split=2, max_depth=6)
    my_rfr.fit(X_train, y_train)
    my_rfr_prediction = my_rfr.predict(X_test)
    testing_model.test(y_test, my_rfr_prediction, "My Random Forest")
    # Display feature importances
    print("Feature importances:", my_rfr.feature_importances_)

    # Sort and display the ranked features
    sorted_indices = np.argsort(my_rfr.feature_importances_)[::-1]
    print("Ranked features:")
    for i, index in enumerate(sorted_indices):
        print(f"Feature {index}: {my_rfr.feature_importances_[index]}")
    # my_rfr.plot_validation_scores()
    # Normalize the feature importance's
    normalized_importances = my_rfr.feature_importances_ / np.sum(my_rfr.feature_importances_)

    # Sort and display the ranked features with normalized importance's
    sorted_indices = np.argsort(normalized_importances)[::-1]
    print("Ranked features with normalized importances:")
    for i, index in enumerate(sorted_indices):
        print(f"Feature {index}: {normalized_importances[index]}")

    # Plot the normalized feature importance's
    plt.figure(figsize=(12, 6))
    plt.bar(range(X_test.shape[1]), normalized_importances[sorted_indices], align='center')
    plt.xticks(range(X_test.shape[1]), sorted_indices)
    plt.xlabel('Feature Index')
    plt.ylabel('Normalized Importance')
    plt.title('Feature Importances')
    plt.show()


def main():
    # Importing dataset
    kc_dataset = pd.read_csv(r'./Data/kc_house_data.csv')

    kc_dataset = kc_dataset.drop(['yr_renovated', 'sqft_basement', 'floors', 'condition'], axis=1)

    # feature indices
    for i, index in enumerate(kc_dataset.columns):
        print(f"Feature {i} == {index}")
    # Splitting train test data
    X_train, X_test, y_train, y_test = data_preprocess.preprocess(kc_dataset)

    # Giving train test data to built-in sklearn models
    sklearnModelsResults(X_train, X_test, y_train, y_test)

    # Giving train test data to my models
    myModelsResults(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
