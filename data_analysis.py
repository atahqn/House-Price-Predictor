import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math


def corr_analysis(dataset: pd.DataFrame):
    isNull = dataset.isnull().values.any()
    print("Dataset does not contain any null values") if (isNull == False) else print("Dataset contains null values")

    plt.figure(figsize=(20, 17))
    matrix = np.triu(dataset.corr())
    sns.heatmap(dataset.corr(), annot=True,
                linewidth=.8, mask=matrix)
    plt.yticks(rotation=0)
    plt.show()
    return dataset


def plotOfFeatures(dataset, features: list, plot_type: str):
    num_features = len(features)
    num_rows = math.ceil(num_features / 5)  # Calculate the required number of rows

    figure, axes = plt.subplots(num_rows, 5, figsize=(10, 10 * num_rows / 5))
    plt.tight_layout(pad=2.0)
    # print(num_features)
    if plot_type == "histogram":
        for i in range(num_features):
            row = i // 5
            col = i % 5
            sns.histplot(dataset[features[i]], ax=axes[row, col])
            # plt.xlabel(features[i])

    elif plot_type == "distribution":
        for i in range(num_features):
            row = i // 5
            col = i % 5
            sns.kdeplot(dataset[features[i]], ax=axes[row, col])
            # plt.xlabel(features[i])
    else:
        print("plot type is invalid")
    # Deleting last two empty plots
    plt.delaxes(axes[3, 3])
    plt.delaxes(axes[3, 4])
    plt.show()


def getFeatureList(dataset):
    return dataset.columns.tolist()


def main():
    # Importing dataset
    kc_dataset = pd.read_csv(r'./Data/kc_house_data.csv')
    # Dropping unnecessary columns
    dataset = kc_dataset.drop(['id', 'date', 'price'], axis=1)

    # Arranging boundaries for display of rows columns and width in pandas
    pd.set_option('display.max_rows', 50)
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.width', 1000)

    # Getting correlation data
    data = corr_analysis(dataset)

    # Showing Some important features of dataset
    print(data.describe().transpose())

    # Getting features as list format
    features = getFeatureList(dataset)

    # Getting target as list
    target = ['price']

    # Displaying histogram plots for features
    plotOfFeatures(dataset, features, "histogram")

    # Displaying kde distribution plots for features
    plotOfFeatures(dataset, features, "distribution")

    # Displaying histogram plot for target
    sns.histplot(kc_dataset[target])
    plt.show()

    # Displaying kde distribution plot for target
    sns.kdeplot(kc_dataset[target])
    plt.show()


if __name__ == "__main__":
    main()
