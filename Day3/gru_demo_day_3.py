import torch
import torch.nn as nn

class SequenceModel(nn.Module):
    def __init__(self, input_sz, hidden_sz, model_type='GRU'):
        super().__init__()
        self.hidden_sz = hidden_sz
        self.model_type = model_type
        
        if model_type == 'LSTM':
            self.rnn = nn.LSTM(input_sz, hidden_sz, batch_first=True)
        else:
            self.rnn = nn.GRU(input_sz, hidden_sz, batch_first=True)
            
        self.fc = nn.Linear(hidden_sz, 1)

    def forward(self, x):
        # LSTM returns (out, (h, c)); GRU returns (out, h)
        if self.model_type == 'LSTM':
            out, (h, c) = self.rnn(x)
        else:
            out, h = self.rnn(x)
            
        return self.fc(out[:, -1, :])

# 5-day window, 1 feature (price)
dummy_input = torch.randn(1, 5, 1) 

gru_model = SequenceModel(1, 32, model_type='GRU')
lstm_model = SequenceModel(1, 32, model_type='LSTM')

print(f"GRU Output: {gru_model(dummy_input).item():.4f}")
print(f"LSTM Output: {lstm_model(dummy_input).item():.4f}")
'''
GRU Output: 0.1226
LSTM Output: 0.1670
'''