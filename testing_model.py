import numpy as np


def calculate_r2_score(y_true, y_prediction):
    ss_res = np.sum((y_true - y_prediction) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


def calculate_mse(y_true, y_prediction):
    return np.mean((y_true - y_prediction) ** 2)


def calculate_evs(y_true, y_prediction):
    var_res = np.var(y_true - y_prediction)
    return 1 - var_res / np.var(y_true)


def test(y_test, y_prediction, model_name: str, isReturn=False):
    r2 = calculate_r2_score(y_test, y_prediction)
    mse = calculate_mse(y_test, y_prediction)
    evs = calculate_evs(y_test, y_prediction)

    print("------------------------------------------")
    print(f" Test results for {model_name}")
    print("------------------------------------------")
    print("| R2 Score  | MSE         | EVS Score    |")
    print("|-----------|-------------|--------------|")
    print("| {0: <9.5f} | {1: <11.5f} | {2: <12.5f} |".format(r2, mse, evs))
    print("------------------------------------------")

    if isReturn: return r2, mse, evs


def main():
    pass


if __name__ == "__main__":
    main()
