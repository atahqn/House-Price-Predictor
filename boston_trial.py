import numpy as np
import pandas as pd



df = pd.read_csv(r'./boston/boston.csv')
X = df.drop(columns=['MEDV'], axis=1)
y = df['MEDV']
y=y.to_numpy()
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Performing train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=1 / 4,random_state=42)

from sklearn.metrics import r2_score
import decision_tree
dtr = decision_tree.DecisionTreeRegressor(min_samples_split=2, max_depth=2)
dtr.fit(X_train, y_train)
dtr_pred = dtr.predict(X_test)
# dtr.print_tree()
print("dtr regression r2 score: ", r2_score(y_test, dtr_pred))
