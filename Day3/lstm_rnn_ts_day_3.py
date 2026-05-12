import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# 1. Generate Synthetic Sequential Data (A sine wave with noise)
t = np.linspace(0, 100, 1000)
data = np.sin(t) + np.random.normal(0, 0.1, 1000)
train, test = data[:800], data[800:]

# --- MODEL 1: ARIMA (Classical Statistical) ---
# Good for linear, stationary trends. No "deep" learning.
arima_model = ARIMA(train, order=(5, 1, 0))
arima_fit = arima_model.fit()
arima_pred = arima_fit.forecast(steps=200)

# --- PREP DATA FOR NEURAL NETS ---
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data)-seq_length):
        xs.append(data[i:(i+seq_length)])
        ys.append(data[i+seq_length])
    return torch.tensor(xs).float().unsqueeze(-1), torch.tensor(ys).float()

X_train, y_train = create_sequences(train, 10)

# --- MODEL 2 & 3: RNN vs LSTM Architectures ---
class SeqModel(nn.Module):
    def __init__(self, mode='RNN'):
        super().__init__()
        self.mode = mode
        if mode == 'RNN':
            self.rnn = nn.RNN(input_size=1, hidden_size=32, batch_first=True)
        else:
            self.rnn = nn.LSTM(input_size=1, hidden_size=32, batch_first=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        # RNN returns (out, h_n); LSTM returns (out, (h_n, c_n))
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

# Initializing models
rnn_model = SeqModel(mode='RNN')
lstm_model = SeqModel(mode='LSTM')

# (Training logic omitted for brevity: standard MSELoss and Adam Optimizer)
print("Models initialized for training on sequence length 10...")