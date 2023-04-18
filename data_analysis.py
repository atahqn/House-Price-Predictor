import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def data_analysis(dataset: pd.DataFrame):
    corr_data = dataset.drop(['id', 'date', 'price'], axis=1)
    isNull = dataset.isnull().values.any()
    print("Dataset does not contain any null values") if (isNull == False) else print("Dataset contains null values")

    plt.figure(figsize=(20, 17))
    matrix = np.triu(corr_data.corr())
    sns.heatmap(corr_data.corr(), annot=True,
                linewidth=.8, mask=matrix, cmap="rocket")
    plt.show()


def main():
    kc_dataset = pd.read_csv(r'./Data/kc_house_data.csv')
    data_analysis(kc_dataset)


if __name__ == "__main__":
    main()
