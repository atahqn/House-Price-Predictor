import pandas as pd

import Multi_Layer_Perceptron
import data_preprocess
import Random_Forest_Regressor
import linear_models
import testing_model

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import (Lasso, Ridge)

from sklearn.neural_network import MLPRegressor


def sklearnModelsResults(X_train, X_test, y_train, y_test):
    # Built in linear regression
    slr = LinearRegression()
    slr.fit(X_train, y_train)
    slr_prediction = slr.predict(X_test)
    testing_model.test(y_test, slr_prediction, "Built-in Linear Regression")

    # Built in ridge regression
    rg = Ridge(alpha=0.1)
    rg.fit(X_train, y_train)
    rg_prediction = rg.predict(X_test)
    testing_model.test(y_test, rg_prediction, "Built-in Ridge Regression")

    # Built in lasso regression
    ls = Lasso(alpha=0.0001)
    ls.fit(X_train, y_train)
    ls_prediction = ls.predict(X_test)
    testing_model.test(y_test, ls_prediction, "Built-in Lasso Regression")

    # Built in random forest regression
    sk_rfr = RandomForestRegressor(n_jobs=-1, n_estimators=3, min_samples_split=2, max_depth=6)
    sk_rfr.fit(X_train, y_train)
    sk_rfr_prediction = sk_rfr.predict(X_test)
    testing_model.test(y_test, sk_rfr_prediction, "Built-in Random Forest")

    # Built in mlp regression
    model_MLP = MLPRegressor(solver="adam", hidden_layer_sizes=(16, 8), learning_rate_init=0.05,
                             max_iter=1000, activation="logistic")
    model_MLP.fit(X_train, y_train)
    mlp_prediction = model_MLP.predict(X_test)
    testing_model.test(y_test, mlp_prediction, "Built-in MLP")


def myModelsResults(X_train, X_test, y_train, y_test):
    # Fitting data to my linear regression model
    CustomLinearRegression = linear_models.LinearRegression()
    CustomLinearRegression.fit(X_train, y_train)
    y_prediction = CustomLinearRegression.predict(X_test)
    testing_model.test(y_test, y_prediction, "My Linear Regression")
    CustomLinearRegression.plot_scores_and_losses()

    # Fitting data to my Random Forest Regressor model
    my_rfr = Random_Forest_Regressor.RandomForestRegressor(n_estimators=3, min_samples_split=2, max_depth=6)
    my_rfr.fit(X_train, y_train)
    my_rfr_prediction = my_rfr.predict(X_test)
    testing_model.test(y_test, my_rfr_prediction, "My Random Forest")
    my_rfr.plot_validation_scores()

    # Reshaping labels to fit my mlp
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    # Fitting data to my MLP model
    my_mlp = Multi_Layer_Perceptron.MLPRegressor(hidden_layer_sizes=(16, 8), learning_rate=0.05, max_iter=1000,
                                                 activation="sigmoid", initialization="random", loss_func="mse")
    my_mlp.fit(X_train, y_train)
    my_mlp_prediction = my_mlp.predict(X_test)
    testing_model.test(y_test, my_mlp_prediction, "My MLP results")
    my_mlp.plot_scores_and_losses()


def main():
    # Importing dataset
    kc_dataset = pd.read_csv(r'./Data/kc_house_data.csv')

    # Splitting train test data
    X_train, X_test, y_train, y_test = data_preprocess.preprocess(kc_dataset)

    # Giving train test data to built-in sklearn models
    sklearnModelsResults(X_train, X_test, y_train, y_test)

    # Giving train test data to my models
    myModelsResults(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
