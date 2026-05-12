import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# 1. Load and Prepare Data
# file_path = "Quote-Equity-FRACTAL-EQ-09-11-2025-09-05-2026.csv"
# file_path = "C:\\Users\\Harsheet Gandhi\\Downloads\\Quote-Equity-FRACTAL-EQ-09-11-2025-09-05-2026.csv"
# df = pd.read_csv(file_path)
import pandas as pd

# Load the data and tell pandas to parse commas as thousand separators
df = pd.read_csv('C:\\Users\\Harsheet Gandhi\\Downloads\\Quote-Equity-FRACTAL-EQ-09-11-2025-09-05-2026.csv', thousands=',')

# Now df['CLOSE'] (or whichever column you are using) will be a float
prices = df['CLOSE'] 

# Fit the model
from statsmodels.tsa.arima.model import ARIMA
arima_model = ARIMA(prices, order=(5, 1, 0))

# Clean column names and format dates
df.columns = df.columns.str.strip()
# df['DATE'] = pd.to_datetime(df['DATE'], format='%d-%b-%Y')
# df = df.sort_values('DATE').reset_index(drop=True)
# df.set_index('DATE', inplace=True)

# # Clean CLOSE price (remove commas if any and convert to float)
# if df['CLOSE'].dtype == object:
#     df['CLOSE'] = df['CLOSE'].str.replace(',', '').astype(float)

# prices = df['CLOSE'].values

# # 2. ARIMA Forecast (7 Days)
# arima_model = ARIMA(prices, order=(5, 1, 0)) 
arima_fit = arima_model.fit()
arima_forecast = arima_fit.forecast(steps=7)

# 3. Neural Network Preprocessing (Scaling between 0 and 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(prices.reshape(-1, 1))

# Create sequences of 10 days to predict the 11th
window_size = 10
X, y = [], []
for i in range(len(scaled_data) - window_size):
    X.append(scaled_data[i:i+window_size])
    y.append(scaled_data[i+window_size])

X = torch.tensor(np.array(X), dtype=torch.float32)
y = torch.tensor(np.array(y), dtype=torch.float32)

# 4. Define Architectures
class StockRNN(nn.Module):
    def __init__(self):
        super(StockRNN, self).__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

class StockLSTM(nn.Module):
    def __init__(self):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        return self.fc(out[:, -1, :])

torch.manual_seed(42)
rnn_model = StockRNN()
lstm_model = StockLSTM()

criterion = nn.MSELoss()
rnn_optimizer = torch.optim.Adam(rnn_model.parameters(), lr=0.005)
lstm_optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.005)

# 5. Train Models
epochs = 150
print("Training Neural Networks...")
for epoch in range(epochs):
    # Train RNN
    rnn_model.train()
    rnn_optimizer.zero_grad()
    rnn_loss = criterion(rnn_model(X), y)
    rnn_loss.backward()
    rnn_optimizer.step()
    
    # Train LSTM
    lstm_model.train()
    lstm_optimizer.zero_grad()
    lstm_loss = criterion(lstm_model(X), y)
    lstm_loss.backward()
    lstm_optimizer.step()

# 6. Predict Next 7 Days (Iterative Sliding Window)
rnn_model.eval()
lstm_model.eval()

# Grab the very last 10 days from the historical data
last_window = torch.tensor(scaled_data[-window_size:], dtype=torch.float32).view(1, window_size, 1)

def predict_next_n_days(model, initial_window, n_days=7):
    preds = []
    curr_window = initial_window.clone()
    with torch.no_grad():
        for _ in range(n_days):
            pred = model(curr_window)
            preds.append(pred.item())
            # Slide window: drop oldest day, append the new prediction
            curr_window = torch.cat((curr_window[:, 1:, :], pred.view(1, 1, 1)), dim=1)
    return preds

rnn_preds_scaled = predict_next_n_days(rnn_model, last_window, 7)
lstm_preds_scaled = predict_next_n_days(lstm_model, last_window, 7)

# Inverse transform to get actual INR prices
rnn_forecast = scaler.inverse_transform(np.array(rnn_preds_scaled).reshape(-1, 1)).flatten()
lstm_forecast = scaler.inverse_transform(np.array(lstm_preds_scaled).reshape(-1, 1)).flatten()

# 7. Generate Future Dates (Skip Weekends)
last_date = df.index[-1]
future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=7)

# 8. Visualization
plt.figure(figsize=(14, 7))

# Plot last 60 days of historical data for context
plot_days = 60
plt.plot(df.index[-plot_days:], df['CLOSE'].iloc[-plot_days:], label='Historical Close Price', color='black', linewidth=2, marker='o', markersize=4)

# Plot Forecasts
plt.plot(future_dates, arima_forecast, label='ARIMA Forecast', color='red', linestyle='--', marker='o')
plt.plot(future_dates, rnn_forecast, label='RNN Forecast', color='blue', linestyle='--', marker='x', markersize=8)
plt.plot(future_dates, lstm_forecast, label='LSTM Forecast', color='green', linestyle='--', marker='s')

plt.title("Fractal Analytics: 7-Day Price Forecast (ARIMA vs RNN vs LSTM)")
plt.xlabel("Date")
plt.ylabel("Price (INR)")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("fractal_forecast.png")