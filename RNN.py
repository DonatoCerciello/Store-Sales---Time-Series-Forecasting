# %%
# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tqdm as tqdm
import pickle
import shap
import argparse
import seaborn as sns


from sklearn.preprocessing import MinMaxScaler
import torch
from torch import nn
from torch.utils.data import DataLoader
# %%
# Parser
parser = argparse.ArgumentParser(allow_abbrev=False)

parser.add_argument('--type_data', type=str, default='family', help='Type of data to use: family, store')
parser.add_argument('--kaggle', type=bool, default=False, help='If the data is for kaggle competition')
parser.add_argument('--model_type', type=str, default='LSTM', help='Which RNN we want to run')
parser.add_argument('--type_output', type=str, default='single', help='If we want to consider multioutput (single, multi)')
parser.add_argument('--XAI', type=bool, default=False, help='Explainability through SHAP')
parser.add_argument('--input_length', type=int, default=30, help = 'Length of the input sequence')
parser.add_argument('--horizon', type=int, default=16, help = 'Length of the output sequence')
parser.add_argument('--batch_size', type=int, default=32, help = 'Batch size')
parser.add_argument('--hidden_dim', type=int, default=128, help = 'Hidden dimension')
parser.add_argument('--num_layer', type=int, default=2, help = 'Number of layers')
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

class RNN(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        self.num_classes = model_args['num_classes']  # number of output classes
        self.num_layers = model_args['num_layers']  # number of LSTM layers
        self.cond_size = model_args['cond_size'] # condition feature size
        self.hidden_size = model_args['hidden_dim']  # hidden state size
        self.model_type = model_args['model_type'] # type of RNN
        self.type_output = model_args['type_output']  # type of output (single, multi)

        self.input = self.num_classes + self.cond_size
        
        if self.model_type == 'LSTM':
            self.lstm_layers = nn.ModuleList([
                nn.LSTM(input_size=self.input if i == 0 else self.hidden_size,
                        hidden_size=self.hidden_size,
                        batch_first=True,
                        bidirectional=False,
                        )
                for i in range(self.num_layers)
            ])
        if self.model_type == 'GRU':
            self.gru_layers = nn.ModuleList([
                nn.GRU(
                    input_size=self.input if i == 0 else self.hidden_size,
                    hidden_size=self.hidden_size,
                    batch_first=True,
                    )
                for i in range(self.num_layers)
            ])
        
        self.batchnorm_layers = nn.ModuleList([
                nn.BatchNorm1d(self.hidden_size) 
                for _ in range(self.num_layers)
            ])
        self.dropout = nn.Dropout(0.2)

        # Fully connected layers for final prediction
        self.relu1 = nn.ReLU()  # ReLU activation function
        self.fc_1 = nn.Linear(self.hidden_size, self.hidden_size//2)  # first fully connected layer
        self.relu2 = nn.ReLU()
        self.fc_2 = nn.Linear(self.hidden_size//2, self.num_classes)  # output layer

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        if self.model_type == 'LSTM':
            for lstm_layer in self.lstm_layers:
                for name, param in lstm_layer.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param)  # Xavier initialization for input-hidden weights
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param)  # Orthogonal initialization for hidden-hidden weights
                    elif 'bias' in name:
                        nn.init.zeros_(param)  # Inizializza i bias a zero
        else:
            for gru_layers in self.gru_layers:
                for name, param in gru_layers.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param)  # Xavier initialization for input-hidden weights
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param)  # Orthogonal initialization for hidden-hidden weights
                    elif 'bias' in name:
                        nn.init.zeros_(param)  # Inizializza i bias a zero
        nn.init.xavier_uniform_(self.fc_1.weight)
        nn.init.zeros_(self.fc_1.bias)
        nn.init.xavier_uniform_(self.fc_2.weight)
        nn.init.zeros_(self.fc_2.bias)

    def forward(self, x):

        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        output = x
        if self.model_type == 'LSTM':
            for i in range(self.num_layers):
                # Pass input through each LSTM layer
                output, (hn, cn) = self.lstm_layers[i](output, (h_0[[i]], c_0[[i]]))
                
                # Apply batch normalization on the output of each layer
                output = output.contiguous().view(-1, self.hidden_size)  # Flatten to (batch_size * seq_len, hidden_size)
                output = self.batchnorm_layers[i](output)
                output = output.view(x.size(0), -1, self.hidden_size)  # Reshape back to (batch_size, seq_len, hidden_size)
                output = self.dropout(output)
                
        if self.model_type == 'GRU':
            for i in range(self.num_layers):
                # Passa l'input attraverso ciascun layer GRU
                output, hn = self.gru_layers[i](output, h_0[[i]])
                
                # Batch Normalization
                output = output.contiguous().view(-1, self.hidden_size)
                output = self.batchnorm_layers[i](output)
                output = output.view(x.size(0), -1, self.hidden_size)
                output = self.dropout(output)

        # Take the last hidden state from the final layer
        out = output[:, -1, :]  # Last hidden state from the sequence (last time step)

        # Apply fully connected layers
        out = self.relu1(out)  # ReLU activation
        out = self.fc_1(out)  # first dense layer
        out = self.relu2(out)  # ReLU activation
        out = self.fc_2(out)  # final output layer
        
        return out.unsqueeze(1)


def encode_time_features(dates):
    """
    Encode time features in a sinusoidal form.

    Parameters:
    - dates: List of dates.

    Returns:
    - np.array containing the sinusoidal encoding of the dates (sin_week, cos_week, sin_month, cos_month).
    """
    days_of_week = np.array([d.weekday() for d in dates])
    days_of_month = np.array([d.month for d in dates])
    
    sin_week = np.sin(2 * np.pi * days_of_week / 7)
    cos_week = np.cos(2 * np.pi * days_of_week / 7)

    sin_month = np.sin(2 * np.pi * days_of_month / 12)
    cos_month = np.cos(2 * np.pi * days_of_month / 12)

    return np.stack([sin_week, cos_week, sin_month, cos_month], axis=1)


def save_model(model, model_save_path):
    torch.save(model.state_dict(), model_save_path)


def split(train_data, test_data, args, fam, sto=None):
    """
    Splits the dataset into training, validation, and test sets.
    It scales the data, and adds time-related features (month of year, day of week) in sinusoidal form.
    
    Parameters:
    - train_data: DataFrame containing the main time series data.
    - test_data : DataFrame containing conditions for Kaggle challenge
    - args: Arguments for the model and training.
    - fam: Family of the data
    - sto: Store of the data (if args.type_output == 'single')
    
    Returns:
    - data_scaler: The MinMaxScaler used for scaling the data.
    - cond_scaler: The MinMaxScaler used for scaling the conditions.
    - gas_scaler : The MinMaxScaler used for scaling the gas prices.
    - train_loader: Loader containing training data with features.
    - val_loader: Loader containing validation data with features.
    - test_loader: Loader containing test data with features.
    - cond_kaggle : Scaled conditions for Kaggle challenge
    """
    
    if sto:
        data = train_data[fam][train_data[fam].columns[train_data[fam].columns.str.contains('sales')]][f'sales_{sto}']
        cond = train_data[fam][train_data[fam].columns[train_data[fam].columns.str.contains('promo')]][f'promo_{sto}']
        exog = train_data[fam][train_data[fam].columns[train_data[fam].columns.str.contains('dcoilwtico_new|holiday')]]
        cond_test = test_data[fam][test_data[fam].columns[test_data[fam].columns.str.contains('promo')]][f'promo_{sto}']
        exog_test = test_data[fam][test_data[fam].columns[test_data[fam].columns.str.contains('dcoilwtico_new|holiday')]]
    else:
        data = train_data[fam][train_data[fam].columns[train_data[fam].columns.str.contains('sales')]]
        cond = train_data[fam][train_data[fam].columns[train_data[fam].columns.str.contains('promo')]]
        exog = train_data[fam][train_data[fam].columns[train_data[fam].columns.str.contains('dcoilwtico_new|holiday')]]
        cond_test = test_data[fam][test_data[fam].columns[test_data[fam].columns.str.contains('promo')]]
        exog_test = test_data[fam][test_data[fam].columns[test_data[fam].columns.str.contains('dcoilwtico_new|holiday')]]
    
    time_data = encode_time_features(data.index)
    time_data = pd.DataFrame(time_data, index=data.index, columns=['sin_week', 'cos_week', 'sin_month', 'cos_month'])
    exog = exog.join(time_data)
    time_test = encode_time_features(cond_test.index)
    time_test = pd.DataFrame(time_test, index=cond_test.index, columns=['sin_week', 'cos_week', 'sin_month', 'cos_month'])
    exog_test = exog_test.join(time_test)

    # Scaling
    scaler_sales = MinMaxScaler()
    scaler_promo = MinMaxScaler()
    scaler_gas_price = MinMaxScaler()

    if args.type_output == 'single':
        data = pd.DataFrame(scaler_sales.fit_transform(data.values.reshape(-1, 1)).reshape(-1, 1), index=data.index)
        cond = pd.DataFrame(scaler_promo.fit_transform(cond.values.reshape(-1, 1)).reshape(-1, 1), index=cond.index)
        exog = pd.DataFrame(scaler_gas_price.fit_transform(exog), columns=exog.columns, index=exog.index)
        cond_test = pd.DataFrame(scaler_promo.transform(cond_test.values.reshape(-1, 1)).reshape(-1, 1), index=cond_test.index)
        exog_test = pd.DataFrame(scaler_gas_price.transform(exog_test), columns=exog_test.columns, index=exog_test.index)
    if args.type_output == 'multi':
        data = pd.DataFrame(scaler_sales.fit_transform(data), columns=data.columns, index=data.index)
        cond = pd.DataFrame(scaler_promo.fit_transform(cond), columns=cond.columns, index=cond.index)
        exog = pd.DataFrame(scaler_gas_price.fit_transform(exog), columns=exog.columns, index=exog.index)
        cond_test = pd.DataFrame(scaler_promo.transform(cond_test), columns=cond_test.columns, index=cond_test.index)
        exog_test = pd.DataFrame(scaler_gas_price.transform(exog_test), columns=exog_test.columns, index=exog_test.index)
    
    cond_kaggle = pd.concat([cond_test, exog_test], axis=1)

    # Windowing
    X, y, X_cond, y_cond, X_exog, y_exog = [], [], [], [], [], []
    window_size = args.input_length
    for i in range(len(data) - window_size - 1):
        X.append(data.iloc[i:i+window_size].values)
        y.append(data.iloc[i+window_size:i+window_size+1].values)
        X_cond.append(cond.iloc[i:i+window_size].values)
        y_cond.append(cond.iloc[i+window_size:i+window_size+1].values)
        X_exog.append(exog.iloc[i:i+window_size].values)
        y_exog.append(exog.iloc[i+window_size:i+window_size+1].values)

    X, y, X_cond, y_cond, X_exog, y_exog = np.array(X), np.array(y), np.array(X_cond), np.array(y_cond), np.array(X_exog), np.array(y_exog)

    train_size = int(len(X) * args.train_percentage)
    test_size = len(X) - train_size
    train_size = int(train_size * 0.8)
    val_size = len(X) - train_size - test_size
    
    X_train, y_train, X_cond_train, y_cond_train, X_exog_train, y_exog_train = X[:train_size], y[:train_size], X_cond[:train_size], y_cond[:train_size], X_exog[:train_size], y_exog[:train_size]
    X_val, y_val, X_cond_val, y_cond_val, X_exog_val, y_exog_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size], X_cond[train_size:train_size+val_size], y_cond[train_size:train_size+val_size], X_exog[train_size:train_size+val_size], y_exog[train_size:train_size+val_size]
    X_test, y_test, X_cond_test, y_cond_test, X_exog_test, y_exog_test = X[train_size+val_size:], y[train_size+val_size:], X_cond[train_size+val_size:], y_cond[train_size+val_size:], X_exog[train_size+val_size:], y_exog[train_size+val_size:]

    # Data Loader
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32), 
        torch.tensor(X_cond_train, dtype=torch.float32), 
        torch.tensor(X_exog_train, dtype=torch.float32), 
        torch.tensor(y_train, dtype=torch.float32)
        )
    val_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_val, dtype=torch.float32), 
        torch.tensor(X_cond_val, dtype=torch.float32), 
        torch.tensor(X_exog_val, dtype=torch.float32), 
        torch.tensor(y_val, dtype=torch.float32)
        )
    test_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_test, dtype=torch.float32), 
        torch.tensor(X_cond_test, dtype=torch.float32), 
        torch.tensor(X_exog_test, dtype=torch.float32), 
        torch.tensor(y_test, dtype=torch.float32),
        torch.tensor(y_cond_test, dtype=torch.float32),
        torch.tensor(y_exog_test, dtype=torch.float32)
        )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print('Data loaded')
    return scaler_sales, scaler_promo, scaler_gas_price, train_loader, val_loader, test_loader, cond_kaggle


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
            for data, cond, exo, targ in train_dataloader:
                optimizer.zero_grad()
                data, cond, exo, targ = data.to(device), cond.to(device), exo.to(device), targ.to(device)
                total_input = torch.cat([data, cond, exo], dim=-1)
                out = model(total_input)

                loss = criterion(out, targ)

                loss.backward()
                if args.type_output != 'single':
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_dataloader)
            train_losses.append(train_loss)

            model.eval()
            val_loss = 0.0
            val_pred = []
            val_true = []
            if args.type_output == 'single':
                save_model(model, model_save_dir + f"/model_last_{sto}.pth")
            else:
                save_model(model, model_save_dir + "/model_last.pth")
            with torch.no_grad():
                for data, cond, exo, targ in val_dataloader:    
                    data, cond, exo, targ = data.to(device), cond.to(device), exo.to(device), targ.to(device)
                    total_input = torch.cat([data, cond, exo], dim=-1)

                    if args.type_output == 'single':
                        out = model(total_input)
                    else:
                        # out = model(data, torch.cat([cond, exo], dim=-1))
                        # out = model(torch.cat([data, cond], dim=-1), exo)
                        out = model(total_input)

                    loss = criterion(out, targ)

                    val_loss += loss.item()
                    val_pred.append(out.cpu().numpy())
                    val_true.append(targ.cpu().numpy())

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
                    if args.type_output == 'single':
                        model_save_path_best = model_save_dir + f"/model_best_{sto}.pth"
                    else:
                        model_save_path_best = model_save_dir + "/model_best.pth"
                    save_model(model, model_save_path_best)
            # Early stop
            if early_stop and validate_score_non_decrease_count >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    except KeyboardInterrupt:
        print("Interrupted")
        if args.type_output == 'single':
            save_model(model, model_save_dir + f"/model_last_{sto}.pth")
        else:
            save_model(model, model_save_dir + "/model_last.pth")

    return train_losses, val_losses, best_epoch, model_best


def predict(model, test_dataloader, args):
    
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for data, cond, exo, targ, targ_cond, targ_exo in test_dataloader:    
            data, cond, exo, targ = data.to(device), cond.to(device), exo.to(device), targ.to(device)
            total_input = torch.cat([data, cond, exo], dim=-1)
            out = model(total_input)

            predictions.append(out.cpu().numpy())
            actuals.append(targ.cpu().numpy())
    
    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)
    return predictions, actuals


def predict_kaggle(model_best, test_dataloader, test_cond, args):
    timestamp_test = pd.date_range(start=cond_kaggle.index.min(), end=cond_kaggle.index.max())
    last_batch = list(test_dataloader)[-1]
    with torch.no_grad():
        model_best.eval()
        data, cond, exo, targ, targ_cond, targ_exo = last_batch[0], last_batch[1], last_batch[2], last_batch[3], last_batch[4], last_batch[5]
        data = data[-1].unsqueeze(0).to(device)
        cond = cond[-1].unsqueeze(0).to(device)
        exo = exo[-1].unsqueeze(0).to(device)
        y = targ[-1].unsqueeze(0).to(device)
        data = torch.cat([data[:, 1:], y], dim=1)
        cond = torch.cat([cond[:, 1:], targ_cond[-1].unsqueeze(0).to(device)], dim=1)
        exo = torch.cat([exo[:, 1:], targ_exo[-1].unsqueeze(0).to(device)], dim=1)
        full_conditions = torch.cat([cond, exo], dim=-1)

        predictions = []
        for t in range(len(timestamp_test)):
            total_input = torch.cat([data, full_conditions], dim=-1)
            output = model(total_input)

            predictions.append(output)
            data = torch.cat([data[:, 1:], output], dim=1)
            promo = test_cond.loc[timestamp_test[t]].values
            promo = torch.Tensor(promo).unsqueeze(0).to(device)
            full_conditions = torch.cat([full_conditions[:, 1:], promo.unsqueeze(1)], dim=1)
        
        predictions = torch.stack(predictions).squeeze().detach().cpu().numpy()
        return predictions
    

# %%

criterion = nn.HuberLoss(reduction="mean")
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')


if args.type_output == 'single':
    for fam in train_data.keys():
        stores = train_data[fam].columns[train_data[fam].columns.str.contains('sales_')]
        df = pd.DataFrame(index=train_data[fam].iloc[-args.horizon:].index, columns=stores)
        df_kaggle = pd.DataFrame(index=test_data[fam].index, columns=stores)
        if args.type_data == 'family':
            stores = [int(store.split('_')[-1]) for store in stores]
        else:
            stores = [str(store.split('_')[-1]) for store in stores]
        for sto in stores:
            if args.type_data == 'store':
                if '/' in sto:
                    sto = sto.replace('/', '_')
            scaler_sales, scaler_promo, scaler_gas_price, train_loader, val_loader, test_loader, cond_kaggle = split(train_data, test_data, args, fam, sto)
            model_args = {
                "num_classes": 1,  # Number of nodes (number of shops)
                "cond_size": 7,  # Number of conditions
                "num_layers": args.num_layer,  # Number of layers
                "hidden_dim" : args.hidden_dim, # Hidden dimension
                "model_type" : args.model_type, # Type of RNN (LSTM, GRU)
                "type_output": args.type_output,  # Type of output (single, multi)
            }

            model = RNN(**model_args).to(device)
            model_save_dir = root_dir + f'/{args.model_type}/{args.type_output}/{fam}_new2'
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

            # Testing
            predictions, real = predict(
                model_best,
                test_loader,
                model_args,
            )
            predictions = scaler_sales.inverse_transform(predictions.reshape(-1, model_args['num_classes'])).reshape(predictions.shape)
            real = scaler_sales.inverse_transform(real.reshape(-1, model_args['num_classes'])).reshape(predictions.shape)
            predictions = predictions[:, 0]
            real = real[:, 0]

            # Save predictions
            df[f'sales_{sto}'] = predictions[-args.horizon:]
            
            if args.XAI:
                # SHAP
                print("SHAP time")
                shap.initjs()
                for i, batch in enumerate(train_loader):
                    if i == 0:
                        input = torch.cat([batch[0], batch[1], batch[2]], dim=-1).to(device)
                    else:
                        input = torch.cat([input, torch.cat([batch[0], batch[1], batch[2]], dim=-1).to(device)], dim=0)
                
                for i, batch in enumerate(val_loader):
                    if i == 0:
                        val = torch.cat([batch[0], batch[1], batch[2]], dim=-1).to(device)
                    else:
                        val = torch.cat([val, torch.cat([batch[0], batch[1], batch[2]], dim=-1).to(device)], dim=0)
                
                explainer = shap.GradientExplainer(model_best, input)
                shap_values = explainer.shap_values(val)

                shap_check = shap_values.mean(axis=0).squeeze(-1)  # From (B, T, input) -> (T, input)
                val_numpy = val.cpu().numpy().mean(axis=0)  # (T, input)
                # summary plot
                shap.summary_plot(shap_check, val_numpy, feature_names=["Sales", "Promo", "oil", "Holiday", "SinWeek", "CosWeek", "SinMonth", "CosMonth"])
                plt.savefig(model_save_dir + f"/shap_summary_{sto}.png")
                plt.close()

                # violin plot
                shap.violin_plot(shap_check, val_numpy, feature_names=["Sales", "Promo", "oil", "Holiday", "SinWeek", "CosWeek", "SinMonth", "CosMonth"], plot_type="violin")
                plt.savefig(model_save_dir + f"/shap_violin_{sto}.png")
                plt.close()

                feature_names = ["Sales", "Promo", "oil", "Holiday", "SinWeek", "CosWeek", "SinMonth", "CosMonth"]

                # Dependence plot
                for feature in feature_names:
                    shap.dependence_plot(feature, shap_check, val_numpy, feature_names=feature_names)
                    plt.savefig(model_save_dir + f"/shap_dependence_{feature}_{sto}.png")
                    plt.close()
                
                # shap bar plot
                shap.summary_plot(shap_check, val_numpy, feature_names=feature_names, plot_type="bar")
                plt.savefig(model_save_dir + f"/shap_bar_{sto}.png")
                plt.close()
                        
                # stacked force plot
                expected_value = model(val).mean().item() 
                shap.plots.force(expected_value, shap_check, feature_names=feature_names)
                plt.savefig(model_save_dir + f"/shap_force_{sto}.png")
                plt.close()

                # heatmap
                plt.figure(figsize=(12, 6))
                sns.heatmap(shap_check.T, cmap="coolwarm", xticklabels=True, yticklabels=feature_names)
                plt.xlabel("Timesteps")
                plt.ylabel("Features")
                plt.title("SHAP Values Heatmap")
                plt.savefig(model_save_dir + f"/shap_heatmap_{sto}.png")
                plt.close()

            if args.kaggle:
                predictions = predict_kaggle(
                    model_best,
                    test_loader,
                    cond_kaggle,
                    args
                )
                predictions = scaler_sales.inverse_transform(predictions.reshape(-1, model_args['num_classes'])).reshape(predictions.shape)
                df_kaggle[f'sales_{sto}'] = predictions

        if args.type_data == 'family' and '/' in fam:
            fam = fam.replace('/', '_')

        save_dir = root_dir + f'/predictions/{args.model_type}/{args.type_output}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        df.to_csv(save_dir + f'/family_{fam}.csv')

        if args.kaggle:
            save_dir = root_dir + f'/kaggle/{args.model_type}/{args.type_output}'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            df_kaggle.to_csv(save_dir + f'/family_{fam}.csv')
else:
    for fam in train_data.keys():
        scaler_sales, scaler_promo, scaler_gas_price, train_loader, val_loader, test_loader, cond_kaggle = split(train_data, test_data, args, fam)
        data_check = next(iter(train_loader))[0]
        model_args = {
            "num_classes": data_check.shape[-1],  # Number of nodes (number of shops)
            "cond_size": data_check.shape[-1] + 6,  # Number of conditions
            "num_layers": args.num_layer,  # Number of layers
            "hidden_dim": args.hidden_dim,  # Hidden dimension
            "model_type": args.model_type,  # Type of RNN (LSTM, GRU)
            "type_output": args.type_output,  # Type of output (single, multi)
        }
        model = RNN(**model_args).to(device)
        model_save_dir = root_dir + f'/{args.model_type}/{args.type_output}/{fam}'
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
        plt.title("Losses for store")
        plt.savefig(model_save_dir + "/losses.png")
        plt.close()

        # Testing
        predictions, real = predict(
            model_best,
            test_loader,
            model_args,
        )
    
        predictions = scaler_sales.inverse_transform(predictions.reshape(-1, model_args['num_classes'])).reshape(predictions.shape)
        real = scaler_sales.inverse_transform(real.reshape(-1, model_args['num_classes'])).reshape(real.shape)
    
        stores = train_data[fam].columns[train_data[fam].columns.str.contains('sales_')]
        df = pd.DataFrame(predictions[-args.horizon:, 0], columns=stores, index=train_data[fam][-args.horizon:].index)
        if args.type_data == 'family' and '/' in fam:
            fam = fam.replace('/', '_')
        
        save_dir = root_dir + f'/predictions/{args.model_type}/{args.type_output}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        df.to_csv(save_dir + f'/family_{fam}.csv')
    
        if args.kaggle:
            predictions = predict_kaggle(
                model_best,
                test_loader,
                cond_kaggle,
                args
            )
            predictions = scaler_sales.inverse_transform(predictions.reshape(-1, model_args['num_classes'])).reshape(predictions.shape)
            df_kaggle = pd.DataFrame(predictions, columns=stores, index=cond_kaggle.index)
            save_dir = root_dir + f'/kaggle/{args.model_type}/{args.type_output}'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            df_kaggle.to_csv(save_dir + f'/family_{fam}.csv')

# %%
