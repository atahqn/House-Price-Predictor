# Kings County House Price Prediction

This project aims to build a model that can predict the house prices in Kings County using the [Kings County House Price dataset](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction), available on Kaggle.

## Overview
The project is divided into four key steps:

1. **Data Preprocessing**
2. **Exploratory Data Analysis (EDA)**
3. **Feature Engineering**
4. **Model Building**

Each step is represented by a Python script.

## Details
### 1. Data Preprocessing (`data_preprocessing.py`)
Data preprocessing involves cleaning the data, handling missing values, removing outliers, and normalizing the features. We load the dataset using pandas and perform various preprocessing tasks. 

### 2. Exploratory Data Analysis (`exploratory_data_analysis.py`)
The EDA process involves visualizing the distribution of various features and their relationship with the target variable. We use matplotlib and seaborn for visualizations to understand the correlations between different features and the target variable.

### 3. Feature Engineering (`feature_engineering.py`)
In this step, we generate new features from existing ones or modify existing features to improve the model's performance.

### 4. Modeling (`model.py`)
Here we define, train, and evaluate our machine learning model. We use Scikit-learn library to import machine learning algorithm, split the data into training and testing sets, fit the model on the training data, make predictions on the test data, and finally, evaluate the model's performance.
