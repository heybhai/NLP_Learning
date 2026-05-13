import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. THE ENCODER ---
class EncoderRNN(nn.Module):
    def __init__(self, input_vocab_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        
        # Maps word indices (e.g., 42) to dense vectors (e.g., 256-D)
        self.embedding = nn.Embedding(input_vocab_size, hidden_size)
        
        # The GRU processes the sequence
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, input_seq):
        # input_seq shape: [batch_size, sequence_length]
        embedded = self.embedding(input_seq)
        
        # output contains all hidden states, 'hidden' is just the final one
        output, hidden = self.gru(embedded)
        
        # We return 'hidden' because this is our CONTEXT VECTOR
        return hidden

# --- 2. THE DECODER ---
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_vocab_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(output_vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        
        # The output layer maps the GRU output back to our vocabulary size
        self.out = nn.Linear(hidden_size, output_vocab_size)

    def forward(self, input_token, hidden):
        # input_token: The previous predicted word [batch_size, 1]
        # hidden: The Context Vector (or the previous hidden state of the decoder)
        
        embedded = self.embedding(input_token)
        embedded = F.relu(embedded)
        
        # The GRU takes the current word and the previous hidden state
        output, hidden = self.gru(embedded, hidden)
        
        # Predict the next word
        prediction = self.out(output)
        
        # Return the prediction and the new hidden state
        return prediction, hidden

# --- 3. TYING IT TOGETHER ---
# Assuming English to French vocabulary sizes
ENG_VOCAB_SIZE = 10000
FRA_VOCAB_SIZE = 12000
HIDDEN_SIZE = 256

encoder = EncoderRNN(ENG_VOCAB_SIZE, HIDDEN_SIZE)
decoder = DecoderRNN(HIDDEN_SIZE, FRA_VOCAB_SIZE)

# Dummy Input: "I love NLP" (Batch Size 1, Seq Length 3)
dummy_input = torch.tensor([[42, 100, 200]])  # Example word indices

print("--- Data Flow ---")
# 1. Pass sentence through Encoder
context_vector = encoder(dummy_input)
print(f"Context Vector Shape: {context_vector.shape} -> (Layers, Batch, Hidden)")

# 2. Setup Decoder starting conditions
# The first token fed to the decoder is always <SOS> (Index 0 usually)
decoder_input = torch.tensor([[0]])  # <SOS> token

# The Decoder uses the Context Vector as its initial memory
decoder_hidden = context_vector

# 3. Take ONE step in the decoder
prediction, next_hidden = decoder(decoder_input, decoder_hidden)
print(f"Prediction Shape: {prediction.shape} -> (Batch, Seq, Target Vocab Size)")

# In a real loop, you would take the argmax of the prediction, 
# and feed that specific word back into the decoder as the next 'decoder_input'