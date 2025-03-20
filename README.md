# Store Sales - Time Series Forecasting

Author: Cerciello Donato

## Abstract
Time series forecasting plays a crucial role in various fields, enabling companies to make informed decisions based on data and optimize their operations. This project addresses the Store Sales - Time Series Forecasting task by implementing a hybrid approach that integrates statistical, machine learning, and deep learning models. Specifically, VAR, ARIMA, SARIMA, and SARIMAX are applied for conventional time series modeling, while Random Forest, XGBoost and Linear Regressor are utilized for machine learning-based forecasting. Furthermore, deep learning methods such as Long Short-Term Memory (LSTM), Gated Recurrent Unit (GRU) and the Spatio-Temporal Informed Dynamic (STID) models are explored. Finally, to improve interpretability, SHAP (SHapley Additive exPlanations) is used to analyze the features importance. Furthermore, a feature masking analysis is conducted for the STID model to assess how conditions with varying levels of similarity impact forecast performance.

The results obtained highlight the strengths and limitations of each approach. Traditional statistical models demonstrate effectiveness in short-term forecasting, but struggle with complex seasonal patterns. In contrast, machine learning models exhibit superior generalization capabilities, while deep learning models, particularly STID, excel at capturing spatio-temporal dependencies. Explainability analysis proves valuable in identifying feature relevance, enhancing both model interpretability and credibility. These findings emphasize the importance of hybrid methodologies and interpretable AI techniques in time series forecasting.


## Data Availability
The dataset used in this project is publicly available from the Store Sales - Time Series Forecasting competition on Kaggle. You can access the dataset and competition details via the following link:
[Kaggle Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)


## Requirements

To set up the environment, create a python environment using the provided `environment.yml` file:  
```sh
conda env create -f environment.yml
conda activate <sales_forecasting>
```

## Preprocess
The preprocessing and data analysis steps are implemented in the [preprocess.ipynb](./preprocess.ipynb) notebook. This notebook includes all the necessary operations for data cleaning, transformation, and feature engineering that were performed on the dataset. It also provides a detailed analysis of the data, visualizations, and insights derived from the initial exploration.


## Machine Learning methods
This script implements various machine learning models for time series forecasting of store sales. It uses models such as ARIMA, SARIMA, SARIMAX, Linear Regression, Random Forest, XGBoost, and VAR. Additionally, it integrates hyperparameter optimization using Optuna.