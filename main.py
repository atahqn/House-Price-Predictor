import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import time

import data_preprocess
import decision_tree
import models
import testing_model


from sklearn.linear_model import LinearRegression as simp_linear_regression
from sklearn.tree import DecisionTreeRegressor as skdtr
from sklearn.ensemble import RandomForestRegressor as sklearnforest
from sklearn.linear_model import (Lasso, Ridge)


def main():
    # Importing dataset
    kc_dataset = pd.read_csv(r'./Data/kc_house_data.csv')

    # Splitting train test data
    X_train, X_test, y_train, y_test = data_preprocess.preprocess(kc_dataset)

    # Fitting data to custom linear regression model
    CustomLinearRegression = models.LinearRegression()
    CustomLinearRegression.fit(X_train, y_train)

    y_pred = CustomLinearRegression.predict(X_test)
    testing_model.test(y_test, y_pred, "My Linear Regression")

    slr = simp_linear_regression()
    slr.fit(X_train, y_train)

    y_pred2 = slr.predict(X_test)

    testing_model.test(y_test, y_pred2, "Built-in Linear Regression")

    rg = Ridge(alpha=0.001)
    rg.fit(X_train, y_train)
    y_pred_rg = rg.predict(X_test)
    testing_model.test(y_test, y_pred_rg, "Built-in Ridge Regression")

    ls = Lasso(alpha=0.0001)
    ls.fit(X_train, y_train)
    y_pred_ls = ls.predict(X_test)
    testing_model.test(y_test, y_pred_ls, "Built-in Lasso Regression")

    rtr = sklearnforest(n_jobs=-1, n_estimators=3, min_samples_split=2, max_depth=6)
    rtr.fit(X_train, y_train)
    rtr_pred = rtr.predict(X_test)
    testing_model.test(y_test, rtr_pred, "Built-in Random Forest")

    time7 = time.time()
    my_rtr = decision_tree.RandomForestRegressor(n_estimators=3, min_samples_split=2, max_depth=6)
    my_rtr.fit(X_train, y_train)
    time8 = time.time()
    print("time passed= ", str(time8 - time7))
    my_rtr_pred = my_rtr.predict(X_test)
    testing_model.test(y_test, my_rtr_pred, "My Random Forest")


if __name__ == "__main__":
    main()
