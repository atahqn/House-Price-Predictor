from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score


def test(y_test, y_pred, model_name: str, isReturn=False):
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred)
    print("--------------------------------------")
    print(f"Results for {model_name}")
    print(f"R2 Score of the model is {r2}")
    print(f"MSE of the model is {mse}")
    print(f"EVS Score of the model is {evs}")
    print("--------------------------------------")
    if isReturn: return r2, mse, evs


def main():
    pass


if __name__ == "__main__":
    main()
