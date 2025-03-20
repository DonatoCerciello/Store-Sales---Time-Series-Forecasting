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


## Command machine_learning.py:
-   type_data : Type of data to use. Choose between family and store. (default : 'family')
-   model : Choose the model for forecasting between arima, sarima, sarimax, linear, random_forest, xgboost, var. (default : 'linear')
-   forecast_days : Number of days to forecast. (default = 16)
-   kaggle : If the forecasting is intended for a Kaggle competition. (default = False)
-   nfits : Number of fits for auto_arima. (default = 50)
-   ntrials : Number of trials for optuna (default = 100)


## Example
```sh
python machine_learning.py --type_data family --model xgboost --forecast_days 16 --kaggle False --ntrials 100
```


## Deep Learning methods
The following scripts implement deep learning models for time series forecasting of store sales. Specifically, the following models are considered: RNN (LSTM, GRU) and [STID](https://arxiv.org/abs/2208.05233)


## Command RNN.py
-   type_data : Type of data to use. Choose between family and store. (default : 'family')
-   model_type : Choose the deep learning model to use between LSTM and GRU. (default : 'LSTM')
-   kaggle : If the forecasting is intended for a Kaggle competition. (default = False)
-   type_output : Specify if the output should be single or multi-output. (default = 'single')
-   XAI : Set this flag to enable explainability through SHAP. (default = False)
-   input_length : Length of the input sequence. (default = 30)
-   horizon : Length of the output sequence to forecast. (default = 16)
-   num_epochs  : Number of epochs for training. (default = 1000)
-   early_stop : Set this flag to use early stopping. (default = True)
-   device : Specify the device to use for training. (default = 'cuda:2')


## Example
```sh
python RNN.py --type_data store --model_type LSTM --XAI True --horizon 16 --batch_size 32 --num_epochs 1000 --learning_rate 1e-4 --device cuda:2
```


## Command STID.py
-   type_data : Type of data to use. Choose between family and store. (default : 'family')
-   STL_check : Set this flag to use STL decompostion. (default : 'True')
-   kaggle : If the forecasting is intended for a Kaggle competition. (default = False)
-   type_output : Specify if the output should be single or multi-output. (default = 'single')
-   XAI : Set this flag to enable explainability through SHAP. (default = False)
-   input_length : Length of the input sequence. (default = 120)
-   horizon : Length of the output sequence to forecast. (default = 16)
-   if_T_i_D : Set this flag to use monthly temporal embedding. (default = True)
-   if_D_i_W : Set this flag to use weekly temporal embedding. (default = True)
-   if_node : Set this flag to use spatial embedding. (default =   True)
-   if_exog : Set this flag to use exogenous features. (default = True)
-   num_epochs  : Number of epochs for training. (default = 1000)
-   early_stop : Set this flag to use early stopping. (default = True)
-   device : Specify the device to use for training. (default = 'cuda:2')


## Example
```sh
python STID.py --type_data store --STL_check True --input_length 120 --horizon 16 --batch_size 32 --num_epochs 1000 --learning_rate 1e-4 --device cuda:2
```