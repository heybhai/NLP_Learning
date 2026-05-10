from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn

# ---------------------------------------------------------
# 1. Generate 200 days of synthetic data
# ---------------------------------------------------------
np.random.seed(42)
torch.manual_seed(42) # Set torch seed for reproducibility
days = 200
prices = [100]
for _ in range(days):
    prices.append(prices[-1] * (1 + np.random.normal(0.001, 0.01)))

df = pd.DataFrame({'Price': prices}, index=pd.date_range(start='2024-01-01', periods=days + 1))

# ---------------------------------------------------------
# 2. ARIMA Model (Statistical Baseline)
# ---------------------------------------------------------
# Fit ARIMA (1,1,1)
arima_model = ARIMA(df['Price'], order=(1, 1, 1))
arima_result = arima_model.fit()

# Forecast the next day
arima_forecast = arima_result.get_forecast(steps=1)
arima_pred_val = arima_forecast.predicted_mean.values[0]
print(f"ARIMA Next Day Prediction: {arima_pred_val:.2f}")

# ---------------------------------------------------------
# 3. Neural Network Preprocessing
# ---------------------------------------------------------
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[['Price']].values)

def create_sequences(data, window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window])
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.float32)

window_size = 5
X, y = create_sequences(scaled_data, window_size)

# ---------------------------------------------------------
# 4. Define Architectures: RNN vs LSTM
# ---------------------------------------------------------
class StockRNN(nn.Module):
    def __init__(self):
        super(StockRNN, self).__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=32, batch_first=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :]) # Output of the last time step

class StockLSTM(nn.Module):
    def __init__(self):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=32, batch_first=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        # LSTM returns output, (hidden_state, cell_state)
        out, (h_n, c_n) = self.lstm(x)
        return self.fc(out[:, -1, :]) # Output of the last time step

# ---------------------------------------------------------
# 5. Initialization & Training
# ---------------------------------------------------------
rnn_model = StockRNN()
lstm_model = StockLSTM()

criterion = nn.MSELoss()
rnn_optimizer = torch.optim.Adam(rnn_model.parameters(), lr=0.01)
lstm_optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.01)

print("Training Neural Networks...")
epochs = 100
for epoch in range(epochs):
    # Train RNN
    rnn_model.train()
    rnn_optimizer.zero_grad()
    rnn_output = rnn_model(X)
    rnn_loss = criterion(rnn_output, y)
    rnn_loss.backward()
    rnn_optimizer.step()
    
    # Train LSTM
    lstm_model.train()
    lstm_optimizer.zero_grad()
    lstm_output = lstm_model(X)
    lstm_loss = criterion(lstm_output, y)
    lstm_loss.backward()
    lstm_optimizer.step()

print(f"Final RNN Loss: {rnn_loss.item():.6f}")
print(f"Final LSTM Loss: {lstm_loss.item():.6f}")

# ---------------------------------------------------------
# 6. Predict Next Day
# ---------------------------------------------------------
rnn_model.eval()
lstm_model.eval()

# Prepare the last 5 days for prediction
last_window = torch.tensor(scaled_data[-window_size:], dtype=torch.float32).view(1, window_size, 1)

with torch.no_grad():
    rnn_pred_scaled = rnn_model(last_window)
    lstm_pred_scaled = lstm_model(last_window)

# Inverse transform to get actual prices
rnn_forecast = scaler.inverse_transform(rnn_pred_scaled.numpy())[0][0]
lstm_forecast = scaler.inverse_transform(lstm_pred_scaled.numpy())[0][0]

print(f"RNN Next Day Prediction:   {rnn_forecast:.2f}")
print(f"LSTM Next Day Prediction:  {lstm_forecast:.2f}")

# ---------------------------------------------------------
# 7. Visualization
# ---------------------------------------------------------
next_day = df.index[-1] + pd.Timedelta(days=1)

plt.figure(figsize=(12, 6))

# Plot the historical data (last 50 days to zoom in on the predictions)
zoom_days = 50
plt.plot(df.index[-zoom_days:], df['Price'].iloc[-zoom_days:], label='Historical Data', color='black', linewidth=1.5)

# Plot ARIMA Prediction
plt.scatter(next_day, arima_pred_val, color='red', marker='o', s=100, zorder=5, 
            label=f'ARIMA: {arima_pred_val:.2f}')

# Plot RNN Prediction
plt.scatter(next_day, rnn_forecast, color='blue', marker='x', s=100, linewidths=3, zorder=5, 
            label=f'RNN: {rnn_forecast:.2f}')

# Plot LSTM Prediction
plt.scatter(next_day, lstm_forecast, color='green', marker='*', s=150, zorder=5, 
            label=f'LSTM: {lstm_forecast:.2f}')

# Formatting the chart
plt.title("ARIMA vs RNN vs LSTM: Next Day Stock Price Prediction")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()