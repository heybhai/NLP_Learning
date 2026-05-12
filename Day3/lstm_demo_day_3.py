import torch
import torch.nn as nn

class ManualLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ManualLSTMCell, self).__init__()
        # Creating linear layers for the 3 gates and 1 candidate memory
        self.W_f = nn.Linear(input_size, hidden_size, bias=False)
        self.U_f = nn.Linear(hidden_size, hidden_size, bias=True)
        
        self.W_i = nn.Linear(input_size, hidden_size, bias=False)
        self.U_i = nn.Linear(hidden_size, hidden_size, bias=True)
        
        self.W_c = nn.Linear(input_size, hidden_size, bias=False)
        self.U_c = nn.Linear(hidden_size, hidden_size, bias=True)
        
        self.W_o = nn.Linear(input_size, hidden_size, bias=False)
        self.U_o = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x, h_prev, c_prev):
        # 1. THE FORGET GATE (Sigmoid)
        f_t = torch.sigmoid(self.W_f(x) + self.U_f(h_prev))
        
        # 2. THE INPUT GATE (Sigmoid) & CANDIDATE MEMORY (Tanh)
        i_t = torch.sigmoid(self.W_i(x) + self.U_i(h_prev))
        c_tilde_t = torch.tanh(self.W_c(x) + self.U_c(h_prev))
        
        # 3. UPDATE THE CELL STATE (The "Memory" Belt)
        forgotten_memory = f_t * c_prev
        new_memory = i_t * c_tilde_t
        c_t = forgotten_memory + new_memory
        
        # 4. THE OUTPUT GATE (Sigmoid)
        o_t = torch.sigmoid(self.W_o(x) + self.U_o(h_prev))
        h_t = o_t * torch.tanh(c_t)
        
        return f_t, i_t, c_tilde_t, c_t, o_t, h_t

# Instantiate and setup initial values
cell = ManualLSTMCell(input_size=1, hidden_size=1)
x_t = torch.tensor([[1.0]])       # New Input token
h_prev = torch.tensor([[0.5]])    # Previous short-term memory
c_prev = torch.tensor([[100.0]])  # Previous long-term memory (Big value)

# Forward pass
f_t, i_t, c_tilde_t, c_t, o_t, h_t = cell(x_t, h_prev, c_prev)
print("Forget Gate (f_t):", f_t.item())
print("Input Gate (i_t):", i_t.item())  
print("Candidate Memory (c_tilde_t):", c_tilde_t.item())
print("Updated Cell State (c_t):", c_t.item())  
print("Output Gate (o_t):", o_t.item())  
print("Updated Long-Term Memory (h_t):", h_t.item())    
