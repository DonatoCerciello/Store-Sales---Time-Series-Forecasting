# %%
# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tqdm as tqdm
import pickle
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import STL

# %%
# Parser
parser = argparse.ArgumentParser(allow_abbrev=False)

parser.add_argument('--type_data', type=str, default='family', help='Type of data to use: family, store')
parser.add_argument('--kaggle', type=bool, default=False, help='If the data is for kaggle competition')
parser.add_argument('--STL_check', type=bool, default=True, help='If we want to use STL decomposition')
parser.add_argument('--input_length', type=int, default=120, help = 'Length of the input sequence')
parser.add_argument('--input_dim', type=int, default=2, help = 'Input dimension (sales, promo)')
parser.add_argument('--exogenous_dim', type=int, default=2, help = 'Exogenous dimension (oil price, holiday)')
parser.add_argument('--horizon', type=int, default=16, help = 'Length of the output sequence')
parser.add_argument('--batch_size', type=int, default=32, help = 'Batch size')
parser.add_argument('--embed_dim', type=int, default=256, help = 'Embedding dimension')
parser.add_argument('--num_layer', type=int, default=1, help = 'Number of MLP layers')
parser.add_argument('--node_dim', type=int, default=16, help = 'Spatial embedding dimension')
parser.add_argument('--temp_dim_tid', type=int, default=8, help = 'Monthly temporal embedding dimension')
parser.add_argument('--temp_dim_diw', type=int, default=8, help = 'Weekly temporal embedding dimension')
parser.add_argument('--time_of_day_size', type=int, default=12, help = 'Number of month in a year')
parser.add_argument('--day_of_week_size', type=int, default=7, help = 'Number of days in a week')
parser.add_argument('--if_T_i_D', type=bool, default=True, help = 'Use daily temporal embedding')
parser.add_argument('--if_D_i_W', type=bool, default=True, help = 'Use weekly temporal embedding')
parser.add_argument('--if_node', type=bool, default=True, help = 'Use spatial embedding')
parser.add_argument('--if_exog', type=bool, default=True, help = 'Use exogenous features')
parser.add_argument('--num_epochs', type=int, default=1000, help = 'Number of epochs')
parser.add_argument('--train_percentage', type=float, default=0.9, help = 'Percentage of training data')
parser.add_argument('--learning_rate', type=float, default=1e-4, help = 'Starting learning rate')
parser.add_argument('--decay_rate', type=float, default=0.5, help = 'Decay rate for scheduler')
parser.add_argument('--patience', type=int, default=50, help = 'Patience for early stopping')
parser.add_argument('--lr_update_interval', type=int, default=5, help = 'Interval to update learning rate')
parser.add_argument('--min_lr', type=float, default=1e-5, help = 'Minimum learning rate')
parser.add_argument('--early_stop', type=bool, default=True, help = 'If we want to use early stopping')
parser.add_argument('--device', type=str, default='cuda:2', help = 'Device to use')


args, _ = parser.parse_known_args()
# %%
# Load data
root_dir = os.getcwd()
start_data_dir = root_dir + '/preprocessed_data/'

with open(start_data_dir + 'train_data_' + args.type_data + '.pkl', 'rb') as f:
    train_data = pickle.load(f)

with open(start_data_dir + 'test_data_' + args.type_data + '.pkl', 'rb') as f:
    test_data = pickle.load(f)


if args.type_data == 'store':
    # Delete store 52, because it as only 118 on 1678 days
    train_data.pop(52)
    test_data.pop(52)

# %%
# Models


class MultiLayerPerceptron(nn.Module):

    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(
            in_channels=input_dim,
            out_channels=hidden_dim,
            kernel_size=(1, 1),
            bias=True,
        )
        self.fc2 = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=(1, 1),
            bias=True,
        )
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.15)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))
        hidden = hidden + input_data
        return hidden


class MV_Forecasting(nn.Module):

    def __init__(self, **model_args):
        super().__init__()

        # Attributes
        self.num_nodes = model_args["num_nodes"]
        self.node_dim = model_args["node_dim"]
        self.input_len = model_args["input_len"]
        self.input_dim = model_args["input_dim"]
        self.exogenous_dim = model_args["exogenous_dim"]
        self.embed_dim = model_args["embed_dim"]
        self.output_len = model_args["output_len"]
        self.num_layer = model_args["num_layer"]
        self.temp_dim_tid = model_args["temp_dim_tid"]
        self.temp_dim_diw = model_args["temp_dim_diw"]
        self.time_of_day_size = model_args["time_of_day_size"]
        self.day_of_week_size = model_args["day_of_week_size"]

        self.if_time_in_day = model_args["if_T_i_D"]
        self.if_day_in_week = model_args["if_D_i_W"]
        self.if_spatial = model_args["if_node"]
        self.if_exog = model_args["if_exog"]

        # Spatial embeddings
        if self.if_spatial:
            self.node_emb = nn.Parameter(torch.empty(self.num_nodes, self.node_dim))
            nn.init.xavier_uniform_(self.node_emb)

        # Temporal embeddings
        if self.if_time_in_day:
            self.time_in_day_emb = nn.Parameter(
                torch.empty(self.time_of_day_size, self.temp_dim_tid)
            )
            nn.init.xavier_uniform_(self.time_in_day_emb)

        if self.if_day_in_week:
            self.day_in_week_emb = nn.Parameter(
                torch.empty(self.day_of_week_size, self.temp_dim_diw)
            )
            nn.init.xavier_uniform_(self.day_in_week_emb)

        # Embedding layer
        self.time_series_emb_layer = nn.Conv2d(
            in_channels=self.input_dim * self.input_len,
            out_channels=self.embed_dim,
            kernel_size=(1, 1),
            bias=True,
        )

        # Encoding
        self.hidden_dim = (
            self.embed_dim
            + self.node_dim * int(self.if_spatial)
            + self.temp_dim_tid * int(self.if_day_in_week)
            + self.temp_dim_diw * int(self.if_time_in_day)
        )

        self.encoder = nn.Sequential(
            *[
                MultiLayerPerceptron(self.hidden_dim, self.hidden_dim)
                for _ in range(self.num_layer)
            ]
        )

        # Regression
        self.regression_layer = nn.Conv2d(
            in_channels=self.hidden_dim,
            out_channels=self.output_len,
            kernel_size=(1, 1),
            bias=True,
        )

        # Attention for exogenous features
        if self.if_exog:

            self.exogenous_encoder = nn.Linear(self.exogenous_dim, self.hidden_dim)

            self.attention_layer = nn.MultiheadAttention(
                embed_dim=self.hidden_dim, num_heads=1, batch_first=True
            )

    def forward(
        self,
        history_data: torch.Tensor,
        exogenous_data: torch.tensor,
    ) -> torch.Tensor:

        # Prepare data
        input_data = history_data[..., range(self.input_dim)]

        if self.if_time_in_day:

            t_i_d_data = history_data[..., 2]
            time_in_day_emb = self.time_in_day_emb[
                (t_i_d_data[:, -1, :] * self.time_of_day_size).type(torch.LongTensor)
            ]
        else:
            time_in_day_emb = None

        if self.if_day_in_week:

            d_i_w_data = history_data[..., 3]
            day_in_week_emb = self.day_in_week_emb[
                (d_i_w_data[:, -1, :] * self.day_of_week_size).type(torch.LongTensor)
            ]
        else:
            day_in_week_emb = None

        # Time series embedding
        batch_size, _, num_nodes, input_dim = input_data.shape
        input_data = input_data.transpose(1, 2).contiguous()
        input_data = (
            input_data.view(batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        )
        time_series_emb = self.time_series_emb_layer(input_data)

        node_emb = []

        if self.if_spatial:
            # Expand node embeddings
            node_emb.append(
                self.node_emb.unsqueeze(0)
                .expand(batch_size, -1, -1)
                .transpose(1, 2)
                .unsqueeze(-1)
            )

        # Temporal embeddings
        tem_emb = []
        if time_in_day_emb is not None:
            tem_emb.append(time_in_day_emb.transpose(1, 2).unsqueeze(-1))
        if day_in_week_emb is not None:
            tem_emb.append(day_in_week_emb.transpose(1, 2).unsqueeze(-1))

        hidden = torch.cat([time_series_emb] + node_emb + tem_emb, dim=1)

        # Attention for exogenous data
        if self.if_exog:
            exogenous_encoded = F.relu(self.exogenous_encoder(exogenous_data))

            hidden_transposed = hidden.squeeze(-1).transpose(1, 2)

            attn_output, _ = self.attention_layer(
                hidden_transposed, exogenous_encoded, exogenous_encoded
            )
            hidden = hidden + attn_output.transpose(1, 2).unsqueeze(
                -1
            )

        # Encoding
        hidden = self.encoder(hidden)

        # Regression
        prediction = self.regression_layer(hidden)
        return prediction


class Modelcomplete(nn.Module):

    def __init__(self, model_args):
        super().__init__()

        if model_args['STL_check']:
            self.residual = MV_Forecasting(**model_args)
            self.trend = MV_Forecasting(**model_args)
            self.seasonal = MV_Forecasting(**model_args)
        else:
            self.data = MV_Forecasting(**model_args)

    def forward(
        self, seasonal, residual, trend, exogenous=None
    ) -> torch.Tensor:

        if model_args['STL_check']:
            seasonal = self.seasonal(seasonal, exogenous)
            residual = self.residual(residual, exogenous)
            trend = self.trend(trend, exogenous)

            out = seasonal + residual + trend
        else:
            out = self.data(seasonal, exogenous)
            seasonal = None
            residual = None
            trend = None

        return out, seasonal, residual, trend


def create_sequences_multivariate(series, indices, seq_length, horizon):
    """
    Creates sequences for multivariate time series forecasting.

    Parameters:
    - series: The multivariate time series data.
    - indices: The indices or time steps corresponding to each row in `series`.
    - seq_length: The length of the input sequence.
    - horizon: The number of future time steps to predict.

    Returns:
    - xs: Input sequences as torch tensors.
    - x_indices: Indices corresponding to the input sequences.
    - ys: Output sequences as torch tensors.
    - y_indices: Indices corresponding to the output sequences.
    """

    xs, ys = [], []
    x_indices, y_indices = [], []
    data = series
    for i in range(len(data) - seq_length - horizon + 1):

        x = data[i : i + seq_length, :]  
        y = data[i + seq_length : i + seq_length + horizon, :]
        x_indx = indices[i : i + seq_length]
        y_indx = indices[i + seq_length : i + seq_length + horizon]
        x_indices.append(x_indx)
        y_indices.append(y_indx)
        xs.append(x)
        ys.append(y)

    return (
        torch.tensor(np.array(xs), dtype=torch.float32),
        np.array(x_indices),
        torch.tensor(np.array(ys), dtype=torch.float32),
        np.array(y_indices),
    )


def add_features(
    data, cond, add_cond=None, add_month_of_year=None, add_day_of_week=None
):
    """
    Adds additional time-related features (time of day and day of week) to a given time series dataset.

    Parameters:
    - data: The input time series data.
    - add_day_of_week: Boolean flag to indicate if the 'day of the week' feature should be added.
    - add_month_of_year: Boolean flag to indicate if the 'month of the year' feature should be added.

    Returns:
    - data_with_features: A NumPy array containing the original time series data plus any added time-related features.
    """

    data1 = np.expand_dims(data[0].values, axis=-1)
    cond1 = np.expand_dims(cond[0].values, axis=-1)
    _, n = data[0].shape
    assert data1.shape == cond1.shape
    feature_list = [data1]
    if add_cond:
        feature_list.append(cond1)

    if add_month_of_year:
        # Numerical time_of_month
        month = (data[0].index.month - 1) / 12
        month_tiled = np.tile(month, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(month_tiled)

    if add_day_of_week:
        # Numerical day_of_week
        dow = data[0].index.dayofweek / 7
        dow_tiled = np.tile(dow, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(dow_tiled)

    data_with_features = np.concatenate(feature_list, axis=-1)

    return data_with_features


class TimeSeriesDataset(Dataset):
    def __init__(self, data=None, seasonal=None, trend=None, residual=None, targets=None, exogenous=None, targets_exog=None):

        self.seasonal = seasonal
        self.trend = trend
        self.residual = residual
        self.data = data
        self.targets = targets
        self.exogenous = exogenous
        self.targets_exog = targets_exog

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):

        item = {}
        if self.data is not None:
            item["data"] = self.data[idx]
        if self.seasonal is not None:
            item["seasonal"] = self.seasonal[idx]
        if self.trend is not None:
            item["trend"] = self.trend[idx]
        if self.residual is not None:
            item["residual"] = self.residual[idx]
        if self.targets is not None:
            item["targets"] = self.targets[idx]
        if self.exogenous is not None:
            item["exogenous"] = self.exogenous[idx]
        if self.targets_exog is not None:
            item["targets_exog"] = self.targets_exog[idx] ###########
        return item


def split(data, condition, exog_data, train_percentage, STL_check):
    """
    Splits the dataset into training, validation, and test sets.
    It scales the data, applies STL decomposition, and adds time-related features (month of year, day of week).
    
    Parameters:
    - data: DataFrame containing the main time series data.
    - exog_data: DataFrame containing the exogenous variables.
    - train_percentage: Percentage of training data.

    
    Returns:
    - data_scaler: The MinMaxScaler used for scaling the data.
    - cond_scaler: The MinMaxScaler used for scaling the condition data.
    - exog_scaler: The MinMaxScaler used for scaling the exogenous data.
    - train_data_with_features: Dictionary containing training data with features.
    - val_data_with_features: Dictionary containing validation data with features.
    - test_data_with_features: Dictionary containing test data with features.
    """
    
    # Define the end of the training period based on the train_percentage
    train_end = int(train_percentage * data.shape[0])

    train_data = data.iloc[:train_end]
    test_data = data.iloc[train_end:]
    train_cond = condition.iloc[:train_end]
    test_cond = condition.iloc[train_end:]

    # Scale the training and test data using MinMaxScaler
    data_scaler = MinMaxScaler()
    data_scaled = pd.DataFrame(
        data_scaler.fit_transform(data),
        columns=data.columns,
        index=data.index,
    )

    train_data_scaled = pd.DataFrame(
        data_scaler.transform(train_data),
        columns=train_data.columns,
        index=train_data.index,
    )
    test_data_scaled = pd.DataFrame(
        data_scaler.transform(test_data), columns=test_data.columns, index=test_data.index
    )

    # Scale the cond data
    cond_scaler = MinMaxScaler()
    cond_scaled = pd.DataFrame(
        cond_scaler.fit_transform(condition),
        columns=condition.columns,
        index=condition.index,
    )
    train_cond_scaled = pd.DataFrame(
        cond_scaler.transform(train_cond),
        columns=train_cond.columns,
        index=train_cond.index,
    )
    test_cond_scaled = pd.DataFrame(
        cond_scaler.transform(test_cond),
        columns=test_cond.columns,
        index=test_cond.index,
    )

    # Scale the exogenous data
    train_exog = exog_data.iloc[:train_end]
    test_exog = exog_data.iloc[train_end:]
    exog_scaler = MinMaxScaler()
    train_exog_scaled = pd.DataFrame(
        exog_scaler.fit_transform(train_exog),
        columns=train_exog.columns,
        index=train_exog.index,
    )
    test_exog_scaled = pd.DataFrame(
        exog_scaler.transform(test_exog),
        columns=test_exog.columns,
        index=test_exog.index,
    )
    assert train_data.shape[0] == train_exog.shape[0]
    assert test_data.shape[0] == test_exog.shape[0]
    assert (train_data.index == train_exog.index).all()
    assert (test_data.index == test_exog.index).all()

    # Apply STL decomposition (Seasonal-Trend decomposition using LOESS) to the scaled data
    stl_train_data = pd.DataFrame(columns=data.columns)
    trend_train_data = pd.DataFrame(columns=data.columns)
    seasonal_train_data = pd.DataFrame(columns=data.columns)
    residual_train_data = pd.DataFrame(columns=data.columns)

    stl_test_data = pd.DataFrame(columns=data.columns)
    trend_test_data = pd.DataFrame(columns=data.columns)
    seasonal_test_data = pd.DataFrame(columns=data.columns)
    residual_test_data = pd.DataFrame(columns=data.columns)

    # Perform STL decomposition for each column in the training and test data
    if STL_check:
        for col in train_data.columns:
            result = STL(train_data_scaled[col], seasonal=7).fit()
            stl_train_data[col] = result
            (
                trend_train_data[col],
                seasonal_train_data[col],
                residual_train_data[col],
            ) = (result.trend, result.seasonal, result.resid)

            result = STL(test_data_scaled[col], seasonal=7).fit()
            stl_test_data[col] = result
            (
                trend_test_data[col],
                seasonal_test_data[col],
                residual_test_data[col],
            ) = (result.trend, result.seasonal, result.resid)

        # Split the training data into training and validation sets (80% train, 20% validation)
        train_size = int(0.8 * train_data.shape[0])

        train_index = train_data.index[:train_size]
        val_index = train_data.index[train_size:]
        test_index = test_data.index

        # Prepare data with features
        train_data_1 = {
            "data": train_data_scaled,
            "seasonal": seasonal_train_data,
            "trend": trend_train_data,
            "residual": residual_train_data,
            "index": train_index,
        }
        test_data_1 = {
            "data": test_data_scaled,
            "seasonal": seasonal_test_data,
            "trend": trend_test_data,
            "residual": residual_test_data,
            "index": test_index,
        }

        train_data_with_features = {
            "data": None,
            "seasonal": None,
            "trend": None,
            "residual": None,
            "index": train_index,
        }
        val_data_with_features = {
            "data": None,
            "seasonal": None,
            "trend": None,
            "residual": None,
            "index": val_index,
        }
        test_data_with_features = {
            "data": None,
            "seasonal": None,
            "trend": None,
            "residual": None,
            "index": test_index,
        }

        data_type = list(train_data_with_features.keys())
        data_type.remove("index")
    else:
        train_size = int(0.8 * train_data.shape[0])

        train_index = train_data.index[:train_size]
        val_index = train_data.index[train_size:]
        test_index = test_data.index

        # Prepare data with features
        train_data_1 = {
            "data": train_data_scaled,
            "index": train_index,
        }
        test_data_1 = {
            "data": test_data_scaled,
            "index": test_index,
        }

        train_data_with_features = {
            "data": None,
            "index": train_index,
        }
        val_data_with_features = {
            "data": None,
            "index": val_index,
        }
        test_data_with_features = {
            "data": None,
            "index": test_index,
        }

        data_type = list(train_data_with_features.keys())
        data_type.remove("index")
    
    # Add time-based features (e.g., month of year, day of week) to the data
    for key in data_type:
        train_data_with_features[key] = add_features(
            [train_data_1[key]],
            [train_cond_scaled],
            add_cond=True,
            add_month_of_year=True,
            add_day_of_week=True
        )
        test_data_with_features[key] = add_features(
            [test_data_1[key]],
            [test_cond_scaled],
            add_cond=True, add_month_of_year=True, add_day_of_week=True
        )

        val_data_with_features[key] = train_data_with_features[key][train_size:]
        train_data_with_features[key] = train_data_with_features[key][:train_size]

    # Add exogenous features
    train_data_with_features["exog"] = train_exog_scaled[:train_size].values
    val_data_with_features["exog"] = train_exog_scaled[train_size:].values
    test_data_with_features["exog"] = test_exog_scaled.values

    return data_scaler, cond_scaler, exog_scaler, train_data_with_features, val_data_with_features, test_data_with_features


def create_datasets(train_data_with_features, val_data_with_features, test_data_with_features, args):
    """
    Creates datasets for training, validation, and testing by generating sequences from multivariate time series data.
    
    Parameters:
    - train_data_with_features: Dictionary containing training data with features.
    - val_data_with_features: Dictionary containing validation data with features.
    - test_data_with_features: Dictionary containing test data with features.
    
    Returns:
    - train_dataset: TimeSeriesDataset object for training.
    - val_dataset: TimeSeriesDataset object for validation.
    - test_dataset: TimeSeriesDataset object for testing.
    """
    
    # Initialize dictionaries to store input (X) and output (y) sequences for train, validation, and test sets
    if args.STL_check:
        X_train = {
            "data": None,
            "seasonal": None,
            "trend": None,
            "residual": None,
            "exog": None,
            "indices": None,
        }
        X_val = {
            "data": None,
            "seasonal": None,
            "trend": None,
            "residual": None,
            "exog": None,
            "indices": None,
        }
        X_test = {
            "data": None,
            "seasonal": None,
            "trend": None,
            "residual": None,
            "exog": None,
            "indices": None,
        }

        y_train = {
            "data": None,
            "seasonal": None,
            "trend": None,
            "residual": None,
            "exog": None,
            "indices": None,
        }
        y_val = {
            "data": None,
            "seasonal": None,
            "trend": None,
            "residual": None,
            "exog": None,
            "indices": None,
        }
        y_test = {
            "data": None,
            "seasonal": None,
            "trend": None,
            "residual": None,
            "exog": None,
            "indices": None,
        }
    else:
        X_train = {
            "data": None,
            "exog": None,
            "indices": None,
        }
        X_val = {
            "data": None,
            "exog": None,
            "indices": None,
        }
        X_test = {
            "data": None,
            "exog": None,
            "indices": None,
        }

        y_train = {
            "data": None,
            "exog": None,
            "indices": None,
        }
        y_val = {
            "data": None,
            "exog": None,
            "indices": None,
        }
        y_test = {
            "data": None,
            "exog": None,
            "indices": None,
        }

    # Define the data types that we will process (all keys except "indices")
    data_type = list(X_train.keys())
    data_type.remove("indices")

    # Iterate over each data type and create sequences for train, validation, and test sets
    for key in data_type:
        X_train[key], X_train["indices"], y_train[key], y_train["indices"] = (
            create_sequences_multivariate(
                train_data_with_features[key],
                train_data_with_features["index"],
                args.input_length,
                args.horizon,
            )
        )

        X_val[key], X_val["indices"], y_val[key], y_val["indices"] = (
            create_sequences_multivariate(
                val_data_with_features[key],
                val_data_with_features["index"],
                args.input_length,
                args.horizon,
            )
        )

        X_test[key], X_test["indices"], y_test[key], y_test["indices"] = (
            create_sequences_multivariate(
                test_data_with_features[key],
                test_data_with_features["index"],
                args.input_length,
                args.horizon,
            )
        )

    # Create TimeSeriesDataset objects for train, validation, and test sets
    if args.STL_check:
        train_dataset = TimeSeriesDataset(
            X_train['data'],
            X_train["seasonal"],
            X_train["trend"],
            X_train["residual"],
            y_train["data"],
            X_train["exog"],
        )
        val_dataset = TimeSeriesDataset(
            X_val["data"],
            X_val["seasonal"],
            X_val["trend"],
            X_val["residual"],
            y_val["data"],
            X_val["exog"],
        )
        test_dataset = TimeSeriesDataset(
            X_test["data"],
            X_test["seasonal"],
            X_test["trend"],
            X_test["residual"],
            y_test["data"],
            X_test["exog"],
        )
        test_target = TimeSeriesDataset(
            y_test["data"],
            y_test['seasonal'],
            y_test['trend'],
            y_test['residual'],
            y_test['data'],
            y_test['exog']
        )
    else:
        train_dataset = TimeSeriesDataset(
            X_train['data'],
            None,
            None,
            None,
            y_train["data"],
            X_train["exog"],
        )
        val_dataset = TimeSeriesDataset(
            X_val["data"],
            None,
            None,
            None,
            y_val["data"],
            X_val["exog"],
        )
        test_dataset = TimeSeriesDataset(
            X_test["data"],
            None,
            None,
            None,
            y_test["data"],
            X_test["exog"],
        )
        test_target = TimeSeriesDataset(
            y_test["data"],
            None,
            None,
            None,
            y_test["data"],
            y_test['exog']
        )
    return train_dataset, val_dataset, test_dataset, test_target


def save_model(model, model_save_path):
    torch.save(model.state_dict(), model_save_path)


def train_model(
    train_dataloader,
    val_dataloader,
    criterion,
    device,
    model_save_dir,
    model,
    args,
):
    patience = args.patience
    early_stop = args.early_stop
    num_epochs = args.num_epochs

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    decay_rate = args.decay_rate

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=decay_rate)

    best_epoch = 0
    train_losses = []
    val_losses = []
    min_lr = args.min_lr
    best_validate_loss = np.inf
    validate_score_non_decrease_count = 0

    lr_update_interval = args.lr_update_interval
    try:
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            for batch in train_dataloader:
                optimizer.zero_grad()
                if args.STL_check:
                    seasonal = batch["seasonal"].to(device)
                    trend = batch["trend"].to(device)
                    residual = batch["residual"].to(device)
                else:
                    seasonal = batch['data'].to(device)
                    trend = None
                    residual = None

                exogenous = batch["exogenous"].to(device)

                out, seasonal, residual, trend = model(seasonal, residual, trend, exogenous)

                targets = batch["targets"].to(device)
                targets = targets[..., [0]]

                loss = criterion(out, targets)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_dataloader)
            train_losses.append(train_loss)

            model.eval()
            val_loss = 0.0
            val_pred = []
            val_true = []
            save_model(model, model_save_dir + f"/model_last_{sto}.pth")
            with torch.no_grad():
                for batch in val_dataloader:
                    if args.STL_check:      
                        seasonal = batch["seasonal"].to(device)
                        trend = batch["trend"].to(device)
                        residual = batch["residual"].to(device)
                    else:
                        seasonal = batch['data'].to(device)
                        trend = batch['data'].to(device)
                        residual = batch['data'].to(device)
                    
                    exogenous = batch["exogenous"].to(device)
                    
                    out, seasonal, residual, trend = model(seasonal, residual, trend, exogenous)

                    targets = batch["targets"].to(device)
                    targets = targets[..., [0]]

                    loss = criterion(out, targets)

                    val_loss += loss.item()
                    val_pred.append(out.cpu().numpy())
                    val_true.append(targets.cpu().numpy())

            val_loss /= len(val_dataloader)
            val_losses.append(val_loss)
            print("Epoch: ", epoch, "Train Loss: ", round(train_loss, 5), "Val Loss: ", round(val_loss,5))
            
            if scheduler:
                is_best_for_now = False
                if best_validate_loss > val_loss + 1e-5:
                    best_validate_loss = val_loss
                    is_best_for_now = True
                    validate_score_non_decrease_count = 0
                    model_best = model
                else:
                    validate_score_non_decrease_count += 1
                    
                if (validate_score_non_decrease_count+1) % lr_update_interval == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    if current_lr > min_lr:
                        print(f"Current learning rate: {current_lr}")
                        model.load_state_dict(model_best.state_dict())
                        scheduler.step()
                if is_best_for_now:
                    model_save_path_best = model_save_dir + f"/model_best_{sto}.pth"
                    save_model(model, model_save_path_best)
            # Early stop
            if early_stop and validate_score_non_decrease_count >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    except KeyboardInterrupt:
        print("Interrupted")
        save_model(model, model_save_dir + f"/model_last_{sto}.pth")

    return train_losses, val_losses, best_epoch, model_best


def predict(model, test_dataloader, args):
    
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():   
        for batch in test_dataloader:
            targets = batch["targets"].to(device)
            if args.STL_check:
                seasonal = batch["seasonal"].to(device)
                trend = batch["trend"].to(device)
                residual = batch["residual"].to(device)
            else:
                seasonal = batch['data'].to(device)
                trend = batch['data'].to(device)
                residual = batch['data'].to(device)
            exogenous = batch["exogenous"].to(device)
            
            out, _, _, _ = model(seasonal, residual, trend, exogenous)

            out = out.squeeze(-1)
            targets = targets[..., 0]
            predictions.append(out.cpu().numpy())
            actuals.append(targets.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)

    return predictions, actuals


def predict_kaggle(model, batch, args):
    
    model.eval()
    predictions = []

    with torch.no_grad():
        exogenous = batch["exogenous"].to(device)
        if args.STL_check:
            seasonal = batch["seasonal"].to(device)
            trend = batch["trend"].to(device)
            residual = batch["residual"].to(device)

            out, _, _, _ = model(seasonal, residual, trend, exogenous)
        else:
            data = batch["data"].to(device)
            out, _, _, _ = model(data, None, None, exogenous)

        out = out.squeeze(-1)
        predictions.append(out.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)

    return predictions



# %%
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

# Train model
for sto in train_data.keys():
    data_store = train_data[sto][train_data[sto].columns[train_data[sto].columns.str.contains('sales')]]
    test_store = test_data[sto]
    cond = train_data[sto]
    cond_store = cond[cond.columns[cond.columns.str.contains('promo')]]
    cond_exog = cond[cond.columns[cond.columns.str.contains('dcoilwtico_new|holiday')]]

    data_scaler, cond_scaler, exog_scaler, train_data_with_features, val_data_with_features, test_data_with_features = split(data_store, cond_store, cond_exog, args.train_percentage, args.STL_check)
    train_dataset, val_dataset, test_dataset, target_dataset = create_datasets(train_data_with_features, val_data_with_features, test_data_with_features, args)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    target_loader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=False)

    # Model parameters
    model_args = {
        "num_nodes": data_store.shape[-1],  # Number of nodes (number of shops)
        "node_dim": args.node_dim,  # Spatial embedding dimension
        "input_len": args.input_length,  # Input sequence length
        "input_dim": args.input_dim,  # Input dimension
        "exogenous_dim": args.exogenous_dim,  # Exogenous dimension
        "embed_dim": args.embed_dim,  # Embedding dimension
        "output_len": args.horizon,  # Output sequence length
        "num_layer": args.num_layer,  # Number of MLP layers
        "temp_dim_tid": args.temp_dim_tid,  # Monthly temporal embedding dimension
        "temp_dim_diw": args.temp_dim_diw,  # Weekly temporal embedding dimension
        "time_of_day_size": args.time_of_day_size,  # Number of month in a year
        "day_of_week_size": args.day_of_week_size,  # Number of days in a week
        "if_T_i_D": args.if_T_i_D,  # Use daily temporal embedding
        "if_D_i_W": args.if_D_i_W,  # Use weekly temporal embedding
        "if_exog": args.if_exog,  # Use exogenous features
        "if_node": args.if_node,  # Use spatial embedding
        "STL_check": args.STL_check, # Use STL decomposition,
    }

    model = Modelcomplete(model_args).to(device)
    criterion = nn.HuberLoss(reduction="mean")
    if args.type_data == 'family' and '/' in sto:
        sto = sto.replace('/', '_')
    print("Training model...")
    model_save_dir = root_dir + '/STID'
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    
    dir_name = f'train_monthly{args.if_T_i_D}_weekly{args.if_D_i_W}_exog{args.if_exog}_node{args.if_node}_STL{args.STL_check}'
    model_save_dir = os.path.join(model_save_dir, dir_name)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    train_losses, val_losses, best_epoch, model_best = train_model(
        train_loader,
        val_loader,
        criterion,  
        device,
        model_save_dir,
        model,
        args,
    )

    # Plot losses
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.legend()
    plt.title(f"Losses for store_{sto}")
    plt.savefig(model_save_dir + f"/losses_{sto}.png")
    plt.close()

    # Prediction
    print("Predicting...")
    pred_series, actual_series = predict(model_best, test_loader, args)

    pred_series = np.maximum(pred_series, 0)
    actual_series1 = data_scaler.inverse_transform(
        actual_series.reshape(-1, actual_series.shape[2])
    ).reshape(actual_series.shape)
    pred_series1 = data_scaler.inverse_transform(
        pred_series.reshape(-1, pred_series.shape[2])
    ).reshape(pred_series.shape)

    last = pd.DataFrame(columns=data_store.columns, index=data_store.iloc[-pred_series1.shape[1]:].index)
    for i in range(pred_series1.shape[2]):
        last.iloc[:, i] = pred_series1[-1, :, i]
    
    save_dir = root_dir + f'/predictions/STID/{dir_name}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    last.to_csv(save_dir + f'/family_{sto}' + '.csv')
    
    # Prediction for Kaggle
    if args.kaggle:
        print("Predicting for Kaggle...")
        target_batches = list(target_loader)
        final_target = target_batches[-1]
        test_batches = list(test_loader)
        final_test = test_batches[-1]
        final = {}
        for key in final_target.keys():
            final[key] = torch.cat([final_test[key][[-1]][:, args.horizon:], final_target[key][[-1]]], dim=1)
        pred_series = predict_kaggle(model_best, final, args)
        
        pred_series = np.maximum(pred_series, 0)
        predictions = data_scaler.inverse_transform(
            pred_series.reshape(-1, pred_series.shape[2])
        ).reshape(pred_series.shape)
        final_predictions = pd.DataFrame(columns=data_store.columns, index=test_store.index)
        for i in range(predictions.shape[2]):
            final_predictions.iloc[:, i] = predictions[0, :, i]

        save_kaggle_dir = root_dir + f'/kaggle/STID/{dir_name}'
        if not os.path.exists(save_kaggle_dir):
            os.makedirs(save_kaggle_dir)

        final_predictions.to_csv(save_kaggle_dir + '/family' + sto + '.csv')

# %%
