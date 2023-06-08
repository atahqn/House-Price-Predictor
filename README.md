# Kings County House Price Prediction

This project aims to build models that can predict house prices in Kings County using the [Kings County House Price dataset](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction) available on Kaggle.

## Overview
The project is structured into several key steps, each represented by a separate Python script:

1. **Data Preprocessing** (`data_preprocess.py`)
2. **Data Analysis** (`data_analysis.py`)
3. **Linear Regression Models** (`linear_models.py`)
4. **Multi-Layer Perceptron** (`Multi_Layer_Perceptron.py`)
5. **Random Forest Regressor** (`Random_Forest_Regressor.py`)
6. **Testing Model Performance** (`testing_model.py`)
7. **Main Script** (`main.py`)

## Details
### 1. Data Preprocessing (`data_preprocess.py`)
In this step, the raw data is cleaned and prepared for analysis. The script loads the dataset using pandas and performs various preprocessing tasks, such as handling missing values, removing outliers, and normalizing features. 

### 2. Data Analysis (`data_analysis.py`)
The script is dedicated to exploring the dataset through visualization. It helps in understanding the distribution of various features and analyzing their relationship with the target variable (house prices). It uses libraries like matplotlib and seaborn for visualizations.

### 3. Linear Regression Models (`linear_models.py`)
This script contains the implementation of Linear Regression models using numpy from scratch. The models are trained on the preprocessed data using gradient descent optimization algorithm. Their performance is evaluated based on various metrics such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE) and R-Squared.

### 4. Multi-Layer Perceptron (`Multi_Layer_Perceptron.py`)
This script contains the implementation of a Multi-Layer Perceptron (MLP), which is a type of artificial neural network, using numpy from scratch. The MLP is trained on the preprocessed data using Adam optimizer. The architecture can be customized, including the number of hidden layers, number of neurons in each layer, activation function, learning rate, and other hyperparameters. The MLP's performance is evaluated based on various metrics.

### 5. Random Forest Regressor (`Random_Forest_Regressor.py`)
This script contains the implementation of a Random Forest Regressor using numpy from scratch. The model is trained on the preprocessed data. The number of trees, the depth of the trees, and other hyperparameters can be adjusted. Its performance is evaluated based on various metrics.

### 6. Testing Model Performance (`testing_model.py`)
This script is used to test the performance of all the implemented models using various evaluation metrics. It provides insights into which model performs the best for this prediction task.

### 7. Main Script (`main.py`)
This is the main script that ties all the other scripts together. It orchestrates the entire workflow by calling functions from other scripts in a logical sequence to preprocess the data, perform analysis, train the models, and evaluate their performance.

## Running the Project
1. Install the required Python packages listed in the `requirements.txt` file.
2. Run `main.py` to execute the entire workflow.

## Authors and Licensing
This project is licensed under the MIT license. For more information, see the `LICENSE` file in the project root.

## Acknowledgments
Special thanks to the contributors of the [Kings County House Price dataset](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction) on Kaggle.
