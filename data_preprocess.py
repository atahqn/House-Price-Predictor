import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


def remove_outliers_iqr(data: pd.DataFrame, outlier_threshold: float = 1.5) -> pd.DataFrame:
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1

    # Filter out the outliers
    data_filtered = data[~((data < (Q1 - outlier_threshold * IQR)) | (data > (Q3 + outlier_threshold * IQR))).any(axis=1)]
    return data_filtered


def preprocess(dataset: pd.DataFrame, outlier_removal=False):
    # First dropping unnecessary columns
    dropped_dataset = dataset.drop(['id', 'date'], axis=1)

    if outlier_removal:
        # Outlier detection and removal using IQR method
        dropped_dataset = remove_outliers_iqr(dropped_dataset, outlier_threshold=3)

    # Scaling data
    scaler = preprocessing.MinMaxScaler()
    transformed_data = scaler.fit_transform(dropped_dataset)
    scaled_df = pd.DataFrame(transformed_data, columns=dropped_dataset.columns)
    # print(scaled_df.describe())
    # Separating features and labels
    X = scaled_df.iloc[:, 1:].values
    y = scaled_df.iloc[:, 0].values

    # Shuffling and splitting train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 5, shuffle=True)
    return X_train, X_test, y_train, y_test


def main():
    kc_dataset = pd.read_csv(r'./Data/kc_house_data.csv')
    X_train, X_test, y_train, y_test = preprocess(kc_dataset, True)
    print("Shape of the features of training set: ", X_train.shape)
    print("Shape of the labels of training set: ", y_train.shape)
    print("Shape of the features of test set: ", X_test.shape)
    print("Shape of the labels of test set: ", y_test.shape)


if __name__ == "__main__":
    main()
