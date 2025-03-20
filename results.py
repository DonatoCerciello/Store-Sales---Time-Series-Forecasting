# %%
# Libraries
import pandas as pd
import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

import matplotlib.pyplot as plt
# %%
# Loading Data and Predictions
root_dir = os.getcwd()
print('The root directory is ', root_dir)
path_data = os.path.join(root_dir, 'preprocessed_data', 'train_data_family.pkl')
path_predictions = os.path.join(root_dir, 'predictions')
stid_path = os.path.join(path_predictions, 'STID')
lstm_path = os.path.join(path_predictions, 'LSTM')
gru_path = os.path.join(path_predictions, 'GRU')


def read_results(path):
    results = {}
    for file in os.listdir(path):
        results[file] = pd.read_csv(os.path.join(path, file))
        results[file].set_index('Unnamed: 0', inplace=True)
    return dict(sorted(results.items()))

with open(path_data, 'rb') as f:
    data = pkl.load(f)

files = stid_path + '/train_monthlyTrue_weeklyTrue_exogTrue_nodeTrue_STLTrue'
full_stid = read_results(files)

files = stid_path + '/train_monthlyTrue_weeklyTrue_exogTrue_nodeTrue_STLFalse'
nostl_stid = read_results(files)

files = stid_path + '/train_monthlyTrue_weeklyTrue_exogTrue_nodeFalse_STLTrue'
no_space_stid = read_results(files)

files = stid_path + '/train_monthlyTrue_weeklyTrue_exogFalse_nodeTrue_STLTrue'
no_exog_stid = read_results(files)

files = stid_path + '/train_monthlyTrue_weeklyFalse_exogTrue_nodeTrue_STLTrue'
no_weekly_stid = read_results(files)

files = stid_path + '/train_monthlyFalse_weeklyTrue_exogTrue_nodeTrue_STLTrue'
no_monthly_stid = read_results(files)

files = lstm_path + '/multi'
lstm_multi = read_results(files)

files = lstm_path + '/single'
lstm_single = read_results(files)

files = gru_path + '/multi'
gru_multi = read_results(files)

files = gru_path + '/single'
gru_single = read_results(files)

files = path_predictions + '/arima'
arima = read_results(files)

files = path_predictions + '/sarima'
sarima = read_results(files)

files = path_predictions + '/sarimax'
sarimax = read_results(files)

files = path_predictions + '/var'
var = read_results(files)

files = path_predictions + '/linear'
linear = read_results(files)

files = path_predictions + '/random_forest'
random_forest = read_results(files)

files = path_predictions + '/xgboost'
xgboost = read_results(files)

# %%
# Evaluation Plots

plot_dir = os.path.join(root_dir, 'plots')
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
original = {}
for key in data.keys():
    val = data[key].iloc[-16:][[col for col in data[key].columns if 'sales' in col]].rename_axis('date')
    original[key] = val
    if '/' in key:
        key = key.replace('/', '_')
    full = full_stid[f'family_{key}.csv'] if f'family_{key}.csv' in full_stid.keys() else pd.DataFrame()
    no_monthly = no_monthly_stid[f'family_{key}.csv'] if f'family_{key}.csv' in no_monthly_stid.keys() else pd.DataFrame()
    no_weekly = no_weekly_stid[f'family_{key}.csv'] if f'family_{key}.csv' in no_weekly_stid.keys() else pd.DataFrame()
    no_exog = no_exog_stid[f'family_{key}.csv'] if f'family_{key}.csv' in no_exog_stid.keys() else pd.DataFrame()
    no_space = no_space_stid[f'family_{key}.csv'] if f'family_{key}.csv' in no_space_stid.keys() else pd.DataFrame()
    nostl = nostl_stid[f'family_{key}.csv'] if f'family_{key}.csv' in nostl_stid.keys() else pd.DataFrame()
    
    multi_lstm = lstm_multi[f'family_{key}.csv'] if f'family_{key}.csv' in lstm_multi.keys() else pd.DataFrame()
    single_lstm = lstm_single[f'family_{key}.csv'] if f'family_{key}.csv' in lstm_single.keys() else pd.DataFrame()
    multi_gru = gru_multi[f'family_{key}.csv'] if f'family_{key}.csv' in gru_multi.keys() else pd.DataFrame()
    single_gru = gru_single[f'family_{key}.csv'] if f'family_{key}.csv' in gru_single.keys() else pd.DataFrame()
    
    arima_pred = arima[f'family_{key}.csv'] if f'family_{key}.csv' in arima.keys() else pd.DataFrame()
    sarima_pred = sarima[f'family_{key}.csv'] if f'family_{key}.csv' in sarima.keys() else pd.DataFrame()
    sarimax_pred = sarimax[f'family_{key}.csv'] if f'family_{key}.csv' in sarimax.keys() else pd.DataFrame()
    var_pred = var[f'family_{key}.csv'] if f'family_{key}.csv' in var.keys() else pd.DataFrame()
    linear_pred = linear[f'family_{key}.csv'] if f'family_{key}.csv' in linear.keys() else pd.DataFrame()
    random_forest_pred = random_forest[f'family_{key}.csv'] if f'family_{key}.csv' in random_forest.keys() else pd.DataFrame()
    xgboost_pred = xgboost[f'family_{key}.csv'] if f'family_{key}.csv' in xgboost.keys() else pd.DataFrame()
    methods = [
        full,
        no_monthly,
        no_weekly,
        no_space,
        no_exog,
        nostl,
        multi_lstm,
        single_lstm,
        multi_gru,
        single_gru,
        arima_pred,
        sarima_pred,
        sarimax_pred,
        var_pred,
        linear_pred,
        random_forest_pred,
        xgboost_pred]

    if all(method.empty for method in methods):
        continue
    else:
        stid_save_path = os.path.join(plot_dir, 'STID')
        if not os.path.exists(stid_save_path):
            os.makedirs(stid_save_path)
        plt.figure(figsize=(20, 10))
        plt.plot(val.index, val.sum(axis=1), label="Original Data", color="black")
        plt.plot(val.index, full.sum(axis=1), label="STID_STL", linestyle="dashed", color="blue") if full.shape[1] > 0 else None
        plt.plot(val.index, no_monthly.sum(axis=1), label="STID_nomonth", linestyle="dotted", color="red") if no_monthly.shape[1] > 0 else None
        plt.plot(val.index, no_weekly.sum(axis=1), label="STID_noweek", linestyle="dashdot", color="green") if no_weekly.shape[1] > 0 else None
        plt.plot(val.index, no_space.sum(axis=1), label="STID_nospace", linestyle="dotted", color="purple") if no_space.shape[1] > 0 else None
        plt.plot(val.index, no_exog.sum(axis=1), label="STID_noexog", linestyle="dashdot", color="brown") if no_exog.shape[1] > 0 else None
        plt.plot(val.index, nostl.sum(axis=1), label="STID_noSTL", linestyle="dotted", color="orange") if nostl.shape[1] > 0 else None
        plt.legend(fontsize=15, loc='upper right')
        plt.xticks(fontsize=15, rotation=90)
        plt.yticks(fontsize=15)
        plt.title(f"{key} Forecast vs Actual Data (STID)", fontsize=20, fontweight='bold')
        plt.savefig(f"{stid_save_path}/{key}_forecast_vs_actual_data.png")
        plt.close()

        deep_learning_save_path = os.path.join(plot_dir, 'Deep_Learning')
        if not os.path.exists(deep_learning_save_path):
            os.makedirs(deep_learning_save_path)
        plt.figure(figsize=(20, 10))
        plt.plot(val.index, val.sum(axis=1), label="Original Data", color="black")
        plt.plot(val.index, multi_lstm.sum(axis=1), label="LSTM_multi", linestyle="dashed", color="orange") if multi_lstm.shape[1] > 0 else None
        plt.plot(val.index, single_lstm.sum(axis=1), label="LSTM_single", linestyle="dotted", color="red") if single_lstm.shape[1] > 0 else None
        plt.plot(val.index, multi_gru.sum(axis=1), label="GRU_multi", linestyle="dashdot", color="green")  if multi_gru.shape[1] > 0 else None
        plt.plot(val.index, single_gru.sum(axis=1), label="GRU_single", linestyle="dotted", color="purple") if single_gru.shape[1] > 0 else None
        plt.plot(val.index, full.sum(axis=1), label="STID", linestyle="dotted", color="blue") if full.shape[1] > 0 else None
        plt.xticks(fontsize=15, rotation=90)
        plt.yticks(fontsize=15)
        plt.legend(fontsize=15, loc='upper right')
        plt.title(f"{key} Forecast vs Actual Data (Deep Learning)", fontsize=20, fontweight='bold')
        plt.savefig(f"{deep_learning_save_path}/{key}_forecast_vs_actual_data.png")
        plt.close()

        machine_learning_save_path = os.path.join(plot_dir, 'Machine_Learning')
        if not os.path.exists(machine_learning_save_path):
            os.makedirs(machine_learning_save_path)
        plt.figure(figsize=(20, 10))
        plt.plot(val.index, val.sum(axis=1), label="Original Data", color="black")
        plt.plot(val.index, arima_pred.sum(axis=1), label="ARIMA", linestyle="dashed", color="orange") if arima_pred.shape[1] > 0 else None
        plt.plot(val.index, sarima_pred.sum(axis=1), label="SARIMA", linestyle="dotted", color="pink") if sarima_pred.shape[1] > 0 else None
        plt.plot(val.index, sarimax_pred.sum(axis=1), label="SARIMAX", linestyle="dashdot", color="gold") if sarimax_pred.shape[1] > 0 else None
        plt.plot(val.index, var_pred.sum(axis=1), label="VAR", linestyle="dotted", color="red") if var_pred.shape[1] > 0 else None
        plt.plot(val.index, linear_pred.sum(axis=1), label="Linear", linestyle="dashdot", color="green") if linear_pred.shape[1] > 0 else None
        plt.plot(val.index, random_forest_pred.sum(axis=1), label="Random Forest", linestyle="dotted", color="purple") if random_forest_pred.shape[1] > 0 else None
        plt.plot(val.index, xgboost_pred.sum(axis=1), label="XGBoost", linestyle="dashdot", color="brown") if xgboost_pred.shape[1] > 0 else None
        plt.plot(val.index, full.sum(axis=1), label="STID", linestyle="dotted", color="blue") if full.shape[1] > 0 else None
        plt.xticks(fontsize=15, rotation=90)
        plt.yticks(fontsize=15)
        plt.legend(fontsize=15, loc='upper right')
        plt.title(f"{key} Forecast vs Actual Data", fontsize=20, fontweight='bold')
        plt.savefig(f"{machine_learning_save_path}/{key}_forecast_vs_actual_data.png")
        plt.close()

    for meth in methods:
        for col in val.columns:
            if col not in meth.columns:
                meth[col] = 0
        meth = meth.sort_index(axis=1)

# %%
# Metrics
def smape_loss(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denominator != 0
    diff = np.zeros_like(y_true)
    diff[mask] = np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]
    return 100 * np.mean(diff)


def rmsle_loss(y_true, y_pred):
    return np.sqrt(np.mean(np.square(np.log1p(y_pred) - np.log1p(y_true))))


def evaluate_model(model_name, actual, predicted):
    """
    Evaluates the performance of a prediction model by comparing the actual values 
    with the predicted values using several error metrics. The calculated metrics 
    include rMAE, rMSE, RMSE, SMAPE, R-squared, and RMSLE.

    Parameters:
    -----------
    model_name : str
        The name of the model being evaluated (e.g., 'Model A').
    actual : array-like
        The actual (observed) values of the target variable.
    predicted : array-like
        The predicted values from the model.

    Returns:
    --------
    dict
        A dictionary containing the evaluation metrics for the model:
        "rMAE", "rMSE", "RMSE", "SMAPE", "R-Squared", "RMSLE"."
    """

    mae = 100 * np.mean(np.abs(actual - predicted)) / np.mean(actual)
    mse = 100 * np.mean((actual - predicted) ** 2) / np.mean(actual) ** 2
    rmse = np.mean(np.sqrt(mean_squared_error(actual, predicted)))
    smape = smape_loss(actual, predicted)
    r2 = r2_score(actual, predicted)
    rmsle = rmsle_loss(actual, [x if x >= -0.99 else -0.99 for x in predicted])

    print(f"\n🔍 {model_name} Evaluation:")
    print(f"rMAE: {mae:.4f}")  # Mean Absolute Error
    print(f"rMSE: {mse:.4f}")  # Mean Squared Error
    print(f"rRMSE: {rmse:.4f}")  # Root Mean Squared Error
    print(f"rSMAPE: {smape:.4f}%")  # Symmetric Mean Absolute Percentage Error
    print(f"R-Squared: {r2:.4f}")  # R-Squared
    print(f"RMSLE: {rmsle:.4f}")  # Root Mean Squared Logarithmic Error
    results = {
        "Model": model_name,
        "rMAE": mae,
        "rMSE": mse,
        "RMSE": rmse,
        "SMAPE": smape,
        "R-Squared": r2,
        "RMSLE": rmsle
    }
    return results

def create_list(data):
    """
    Creates a list of numpy arrays from the data dictionary.
    
    Parameters:
    -----------
    data : dict
        A dictionary containing the data to be converted into a list.
        
    Returns:
    --------
    list
        A list of numpy arrays containing the data from the input dictionary.
    """
    all_data = []
    for df in data.values():
        for col in df.columns:
            all_data.append(np.array(df[col].tolist()))
    return np.concatenate(all_data)

def plot_bar_for_model(ax, model_results, index, bar_width, colors, labels_added):
    """
    Helper function to plot the bars for each model's results.
    """
    k = 0
    for key in model_results.keys():
        if key == 'Model':
            name = model_results[key]
        else:
            bars = ax.bar(index[k], model_results[key], bar_width, label=name if not labels_added[name] else "", color=colors)
            # Add the value on top of each bar
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.2f}', ha='center', va='bottom', fontsize=12)
            if not labels_added[name]:
                labels_added[name] = True  # Mark as labeled
            k += 1

def plot_model_comparison(results_list, metrics, save_path):
    """
    Plots a bar chart comparing model evaluation metrics.

    Parameters:
    -----------
    results_list : list
        A list of dictionaries containing the evaluation metrics for each model.
    metrics : list
        A list of metric names to be displayed on the x-axis.
    """

    # Get the model names from the results
    models = []
    for result in results_lists:
        models.append(result['Model'])
    
    # Set up the bar width and colors
    bar_width = 0.12
    index = np.arange(len(metrics))  # positions for each metric

    # Set up the plot
    fig, ax = plt.subplots(figsize=(20, 10))
    colors = list(mcolors.TABLEAU_COLORS.values())  # Custom colors for each model
    labels_added = {model: False for model in models}  # Dictionary to track labeled models

    # Plot bars for each model's results using the helper function
    for i, model_results in enumerate(results_list):
        plot_bar_for_model(ax, model_results, index + i * bar_width - 2.5 * bar_width, bar_width, colors[i], labels_added)

    # Modify the y-axis for better visibility (log scale)
    ax.set_yscale('log')

    # Set the labels, title, and legend
    ax.set_xlabel('Metrics', fontsize=15)
    ax.set_ylabel('Log scale', fontsize=15)
    ax.set_title('Metric comparison', fontsize=20, fontweight='bold')
    ax.set_xticks(index)
    ax.set_xticklabels(metrics)
    ax.legend(title="Models", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=15)

    # Display the plot
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


common_keys = set(full_stid.keys()).intersection(
    set(lstm_multi.keys())).intersection(
    set(lstm_single.keys())).intersection(
    set(gru_multi.keys())).intersection(
    set(gru_single.keys())).intersection(
    set(arima.keys())).intersection(
    set(sarima.keys())).intersection(
    set(sarimax.keys())).intersection(
    set(var.keys())).intersection(
    set(linear.keys())).intersection(
    set(random_forest.keys())).intersection(
    set(xgboost.keys())
)
categories = [file.split('_')[1].replace('.csv', '') for file in common_keys]
if len(common_keys) != 0:
    all_original = create_list({key: original[key] for key in categories})
    all_stid_full = create_list({key: full_stid[key] for key in common_keys})
    all_stid_nomonthly = create_list({key: no_monthly_stid[key] for key in common_keys})
    all_stid_noweekly = create_list({key: no_weekly_stid[key] for key in common_keys})
    all_stid_nospace = create_list({key: no_space_stid[key] for key in common_keys})
    all_stid_noexog = create_list({key: no_exog_stid[key] for key in common_keys})
    all_stid_nostl = create_list({key: nostl_stid[key] for key in common_keys})

    all_lstm_multi = create_list({key: lstm_multi[key] for key in common_keys})
    all_lstm_single = create_list({key: lstm_single[key] for key in common_keys})
    all_gru_multi = create_list({key: gru_multi[key] for key in common_keys})
    all_gru_single = create_list({key: gru_single[key] for key in common_keys})

    all_arima = create_list({key: arima[key] for key in common_keys})
    all_sarima = create_list({key: sarima[key] for key in common_keys})
    all_sarimax = create_list({key: sarimax[key] for key in common_keys})
    all_var = create_list({key: var[key] for key in common_keys})
    all_linear = create_list({key: linear[key] for key in common_keys})
    all_random_forest = create_list({key: random_forest[key] for key in common_keys})
    all_xgboost = create_list({key: xgboost[key] for key in common_keys})

    results = [
        evaluate_model("STID", all_original, all_stid_full),
        evaluate_model("No Monthly", all_original, all_stid_nomonthly),
        evaluate_model("No Weekly", all_original, all_stid_noweekly),
        evaluate_model("No Spatial", all_original, all_stid_nospace),
        evaluate_model("No Exog", all_original, all_stid_noexog),
        evaluate_model("No STL", all_original, all_stid_nostl),
        evaluate_model("LSTM Multi", all_original, all_lstm_multi),
        evaluate_model("LSTM Single", all_original, all_lstm_single),
        evaluate_model("GRU Multi", all_original, all_gru_multi),
        evaluate_model("GRU Single", all_original, all_gru_single),
        evaluate_model("ARIMA", all_original, all_arima),
        evaluate_model("SARIMA", all_original, all_sarima),
        evaluate_model("SARIMAX", all_original, all_sarimax),
        evaluate_model("VAR", all_original, all_var),
        evaluate_model("Linear", all_original, all_linear),
        evaluate_model("Random Forest", all_original, all_random_forest),
        evaluate_model("XGBoost", all_original, all_xgboost)
    ]

    full_results = (results[0])
    nomonthly_results = (results[1])
    noweekly_results = (results[2])
    nospace_results = (results[3])
    noexo_results = (results[4])
    nostl_results = (results[5])
    lstm_multi_results = (results[6])
    lstm_single_results = (results[7])
    gru_multi_results = (results[8])
    gru_single_results = (results[9])
    arima_results = (results[10])
    sarima_results = (results[11])
    sarimax_results = (results[12])
    var_results = (results[13])
    linear_results = (results[14])
    random_forest_results = (results[15])
    xgboost_results = (results[16])

    # STID (comparison)
    results_lists = [full_results, nomonthly_results, noweekly_results, nospace_results, noexo_results, nostl_results]
    metrics = ["rMAE", "rMSE", "RMSE", "SMAPE", "R-Squared", "RMSLE"]
    save_path = os.path.join(stid_save_path, 'STID_comparison.png')
    plot_model_comparison(results_lists, metrics, save_path)

    # Deep Learning (comparison)
    results_lists = [full_results, lstm_multi_results, lstm_single_results, gru_multi_results, gru_single_results]
    metrics = ["rMAE", "rMSE", "RMSE", "SMAPE", "R-Squared", "RMSLE"]
    save_path = os.path.join(deep_learning_save_path, 'Deep_Learning_comparison.png')
    plot_model_comparison(results_lists, metrics, save_path)

    # Machine Learning (comparison)
    results_lists = [full_results, arima_results, sarima_results, sarimax_results, var_results, linear_results, random_forest_results, xgboost_results]
    metrics = ["rMAE", "rMSE", "RMSE", "SMAPE", "R-Squared", "RMSLE"]
    save_path = os.path.join(machine_learning_save_path, 'Machine_Learning_comparison.png')
    plot_model_comparison(results_lists, metrics, save_path)
else:
    raise ValueError("No common keys found between the models: train the models for at least one family/store")

# %%
