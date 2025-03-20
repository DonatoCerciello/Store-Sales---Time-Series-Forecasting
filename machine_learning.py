# %%
# Libraries
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import pickle
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from statsmodels.tsa.api import VAR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import optuna
from sklearn.preprocessing import MinMaxScaler
import warnings
import argparse
warnings.filterwarnings("ignore")

# %%
# Parser
parser = argparse.ArgumentParser(allow_abbrev=False)

parser.add_argument('--type_data', type=str, default='family', help='Type of data to use: family, store')
parser.add_argument('--model', type=str, default='linear', help='Model to use: arima, sarima, sarimax, linear, random_forest, xgboost, var')
parser.add_argument('--forecast_days', type=int, default=16, help='Number of days to forecast')
parser.add_argument('--kaggle', type=bool, default=False, help='If the data is for kaggle competition')
parser.add_argument('--nfits', type=int, default=50, help='Number of fits for auto_arima')
parser.add_argument('--ntrials', type=int, default=100, help='Number of trials for optuna')

args, _ = parser.parse_known_args()
# %%
# Load data
root_dir = os.getcwd()
start_data_dir = root_dir + '/preprocessed_data/'

with open(start_data_dir + 'train_data_' + args.type_data + '.pkl', 'rb') as f:
    data = pickle.load(f)

with open(start_data_dir + 'test_data_' + args.type_data + '.pkl', 'rb') as f:
    test_data = pickle.load(f)

# %%
if args.model not in ['arima', 'sarima', 'sarimax', 'linear', 'random_forest', 'xgboost', 'var']:
    raise ValueError("Model not supported; please choose between arima, sarima, sarimax, linear, random_forest, xgboost, var")

for store in tqdm(data.keys(), desc="Stores"):
    categories = list(set([col.replace("sales_", "").replace("promo_", "") for col in data[store].columns if "sales_" in col or "promo_" in col]))
    if args.type_data == "family":
        categories = [int(cat) for cat in categories]
    categories.sort()
    if args.model == 'var':
        if args.kaggle:
            data_store = data[store][[s for s in data[store].columns if 'sales' in s]]
            test_index = test_data[store].index
        else:
            data_store = data[store][[s for s in data[store].columns if 'sales' in s]].iloc[:-args.forecast_days]
            test_index = data[store].index[-args.forecast_days:]
        var_model = VAR(data_store)
        var_results = var_model.fit(maxlags=7)
        lag_order = var_results.k_ar
        forecast_var = var_results.forecast(data_store.values[-lag_order:], args.forecast_days)
        total = pd.DataFrame(forecast_var, index=test_index, columns=data_store.columns)
    else:
        if args.kaggle:
            total = pd.DataFrame(index=test_data[store].index)
        else:
            total = pd.DataFrame(index=data[store].index[-args.forecast_days:])
        for cat in tqdm(categories, desc="Categories", leave=True, position=1):
            train_input = data[store][f'sales_{cat}']
            if train_input.nunique() == 1:
                print(f"Category {cat} has only one unique value, skipping...")
                continue
            cond_input = data[store][[f'promo_{cat}', 'dcoilwtico_new', 'holiday']]

            start_time = min(train_input[train_input != 0].index)
            train_input = train_input[start_time:]
            cond_input = cond_input[start_time:]

            cond_input['day_sin'] = np.sin(2*np.pi*cond_input.index.dayofweek/7)
            cond_input['day_cos'] = np.cos(2*np.pi*cond_input.index.dayofweek/7)
            cond_input['month_sin'] = np.sin(2*np.pi*cond_input.index.month/12)
            cond_input['month_cos'] = np.cos(2*np.pi*cond_input.index.month/12)
            cond_input['year'] = cond_input.index.year
            if args.kaggle:
                train = train_input
                future_dates = test_data[store].index
                exog = cond_input
                exog_val = test_data[store][[f'promo_{cat}', 'dcoilwtico_new', 'holiday']]
                exog_val['day_sin'] = np.sin(2*np.pi*exog_val.index.dayofweek/7)
                exog_val['day_cos'] = np.cos(2*np.pi*exog_val.index.dayofweek/7)
                exog_val['month_sin'] = np.sin(2*np.pi*exog_val.index.month/12)
                exog_val['month_cos'] = np.cos(2*np.pi*exog_val.index.month/12)
            else:
                val = train_input[-args.forecast_days:]
                train = train_input[:-args.forecast_days]
                future_dates = val.index
                exog = cond_input[:-args.forecast_days]
                exog_val = cond_input[-args.forecast_days:]
            if args.model == 'arima':
                auto_arima_model = auto_arima(train, seasonal=False, suppress_warnings=True, stepwise=True, nfits=args.nfits)
                p, d, q = auto_arima_model.order
                print(f"Best ARIMA order: ({p}, {d}, {q})")
                arima_model = ARIMA(train, order=(p, d, q))
                arima_results = arima_model.fit()
                print(arima_results.summary())

                forecast = arima_results.forecast(steps=args.forecast_days)
                forecast = pd.DataFrame(forecast.values, index=future_dates, columns=[f'sales_{cat}'])
            if args.model == 'sarima':
                auto_sarima_model = auto_arima(train, seasonal=True, m=7, suppress_warnings=True, stepwise=True, nfits=args.nfits)
                p, d, q = auto_sarima_model.order
                P, D, Q, m = auto_sarima_model.seasonal_order
                print(f"Best SARIMA order: ({p}, {d}, {q}) x ({P}, {D}, {Q}, {m})")

                sarima_model = SARIMAX(train, order=(p, d, q), seasonal_order=(P, D, Q, m))
                sarima_results = sarima_model.fit()
                print(sarima_results.summary())

                forecast = sarima_results.forecast(steps=args.forecast_days)
                forecast = pd.DataFrame(forecast.values, index=future_dates, columns=[f'sales_{cat}'])
            if args.model == 'sarimax':
                auto_sarimax_model = auto_arima(train, seasonal=True, m=7, exogenous=exog, suppress_warnings=True, stepwise=True, nfits=args.nfits)
                p, d, q = auto_sarimax_model.order
                P, D, Q, m = auto_sarimax_model.seasonal_order
                print(f"Best SARIMAX order: ({p}, {d}, {q}) x ({P}, {D}, {Q}, {m})")

                sarimax_model = SARIMAX(train, order=(p, d, q), seasonal_order=(P, D, Q, m), exog=exog)
                sarimax_results = sarimax_model.fit()
                print(sarimax_results.summary())

                forecast = sarimax_results.forecast(steps=args.forecast_days, exog=exog_val)
                forecast = pd.DataFrame(forecast.values, index=future_dates, columns=[f'sales_{cat}'])
            if args.model == 'linear':
                minmax = MinMaxScaler()
                train_linear = minmax.fit_transform(train.values.reshape(-1, 1)).flatten()
                train_linear = pd.DataFrame(train_linear, index=train.index, columns=[f'sales_{cat}'])

                cond_minmax = MinMaxScaler()
                exog['dcoilwtico_new_scaled'] = cond_minmax.fit_transform(exog['dcoilwtico_new'].values.reshape(-1, 1)).flatten()
                exog_val['dcoilwtico_new_scaled'] = cond_minmax.transform(exog_val['dcoilwtico_new'].values.reshape(-1, 1)).flatten()
                exog_linear_train = exog[['dcoilwtico_new_scaled', 'holiday', 'day_sin', 'day_cos', 'month_sin', 'month_cos', 'year']]
                exog_linear_val = exog_val[['dcoilwtico_new_scaled', 'holiday', 'day_sin', 'day_cos', 'month_sin', 'month_cos', 'year']]

                rf_model = LinearRegression()
                rf_model.fit(exog_linear_train, train_linear)
                forecast_lr = rf_model.predict(exog_linear_val)
                forecast = pd.DataFrame(minmax.inverse_transform(forecast_lr), index=exog_val.index, columns=[f'sales_{cat}'])
            if args.model == 'random_forest':
                def objective(trial, train=train, exog=exog):
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 200, step=50),
                        'max_depth': trial.suggest_categorical('max_depth', [10, 20, None]),
                        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
                        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
                    }
                    valid_size = int(len(train) * 0.2)
                    exog_train, exog_valid = exog[:-valid_size], exog[-valid_size:]
                    train_data, valid_data = train[:-valid_size], train[-valid_size:]
                    assert len(exog_train) == len(train_data)
                    assert len(exog_valid) == len(valid_data)
                    rf_model = RandomForestRegressor(random_state=42, n_jobs=-1, **params)
                    rf_model.fit(exog_train, train_data)

                    mse = mean_squared_error(valid_data, rf_model.predict(exog_valid))
                    return mse  # Minimize the score
                study = optuna.create_study(direction='minimize')
                study.optimize(objective, n_trials=args.ntrials)
                best_params = study.best_params
                print("Best RF Parameters:", best_params)
                best_rf_model = RandomForestRegressor(random_state=42, n_jobs=-1, **best_params)
                best_rf_model.fit(exog, train)
                rf_predictions = best_rf_model.predict(exog_val)
                forecast = pd.DataFrame(rf_predictions, index=exog_val.index, columns=[f'sales_{cat}'])
            if args.model == 'xgboost':
                def objective(trial, train=train, exog=exog):
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                        'max_depth': trial.suggest_int('max_depth', 3, 20),
                        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
                        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
                        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
                        'gamma': trial.suggest_uniform('gamma', 0, 0.3),
                        'reg_alpha': trial.suggest_uniform('reg_alpha', 0, 0.5),
                        'reg_lambda': trial.suggest_uniform('reg_lambda', 0, 0.5),
                    }
                    valid_size = int(len(train) * 0.2)
                    exog_train, exog_valid = exog[:-valid_size], exog[-valid_size:]
                    train_data, valid_data = train[:-valid_size], train[-valid_size:]
                    assert len(exog_train) == len(train_data)
                    assert len(exog_valid) == len(valid_data)
                    xgb_model = XGBRegressor(tree_method='gpu_hist', **params)
                    xgb_model.fit(exog_train, train_data, eval_set=[(exog_valid, valid_data)], verbose=False)
                    return -xgb_model.score(exog_valid, valid_data)  # Maximizing the R2 score
                study = optuna.create_study(direction='minimize')
                study.optimize(objective, n_trials=args.ntrials)
                best_params = study.best_params
                best_xgb_model = XGBRegressor(tree_method='gpu_hist', **best_params)
                best_xgb_model.fit(exog, train)

                xgb_predictions = best_xgb_model.predict(exog_val)
                forecast = pd.DataFrame(xgb_predictions, index=exog_val.index, columns=[f'sales_{cat}'])
            total = total.join(forecast)
    # Saving the predictions
    if '/' in store:
        store = store.replace('/', '_')
    if not os.path.exists(root_dir + f'/predictions/{args.model}'):
        os.makedirs(root_dir + f'/predictions/{args.model}')
    total.to_csv(root_dir + f'/predictions/{args.model}/{args.type_data}_{store}.csv')
