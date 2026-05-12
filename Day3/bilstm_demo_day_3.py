import torch
import torch.nn as nn

# Parameters
input_size = 10   # e.g., 10-dimensional word embeddings
hidden_size = 20  # internal memory size
num_layers = 1

# Define Bi-LSTM
# Note: bidirectional=True
bi_lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

# Dummy input: (Batch Size, Sequence Length, Input Features)
# 1 sentence, 5 words long, each word is a vector of 10
input_data = torch.randn(1, 5, 10)

# Initializing hidden and cell states
# (num_layers * 2) because we have forward and backward directions
h0 = torch.zeros(num_layers * 2, 1, hidden_size)
c0 = torch.zeros(num_layers * 2, 1, hidden_size)

output, (hn, cn) = bi_lstm(input_data, (h0, c0))

print(f"Input Shape: {input_data.shape}")
# Output shape will be (1, 5, 40) because 20 (hidden) * 2 (directions) = 40
print(f"Output Shape (concatenated): {output.shape}")
print(f"Hidden State Shape: {hn.shape}")
print(f"Cell State Shape: {cn.shape}")  