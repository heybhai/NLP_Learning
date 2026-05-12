import torch
import torch.nn as nn
import torch.optim as optim

# --- STEP 1: DATA PREPARATION ---
data = [
    ("I love this deep learning stuff", 1),
    ("this is the best tutorial ever", 1),
    ("I hate vanishing gradients", 0),
    ("RNNs are so frustrating", 0),
    ("LSTMs make me happy", 1),
    ("the model is failing", 0)
]

# Create Vocabulary
all_words = set([word.lower() for sent, _ in data for word in sent.split()])
word_to_idx = {word: i + 1 for i, word in enumerate(all_words)} # 1-based indexing
word_to_idx["<PAD>"] = 0
vocab_size = len(word_to_idx)

def prepare_sequence(seq, to_ix):
    idxs = [to_ix.get(w.lower(), 0) for w in seq.split()]
    return torch.tensor(idxs, dtype=torch.long)

# --- STEP 2: THE BI-LSTM ARCHITECTURE ---
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(BiLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Bi-directional LSTM
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        
        # Linear layer: hidden_dim * 2 because of bidirectional concatenation
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embeds = self.embedding(x).unsqueeze(0) # [1, seq_len, embed_dim]
        
        # lstm_out: [1, seq_len, hidden_dim * 2]
        # hn: [2, 1, hidden_dim] -> (num_layers * num_directions, batch, hidden)
        lstm_out, (hn, cn) = self.lstm(embeds)
        
        # Concatenate the last hidden state of forward (hn) and backward (hn)
        final_hidden = torch.cat((hn[0,:,:], hn[1,:,:]), dim=1)
        
        return self.sigmoid(self.fc(final_hidden))

# --- STEP 3: TRAINING ---
model = BiLSTMClassifier(vocab_size, embed_dim=16, hidden_dim=32)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCELoss()

print("Training Bi-LSTM...")
for epoch in range(50):
    total_loss = 0
    for sent, label in data:
        model.zero_grad()
        
        sentence_in = prepare_sequence(sent, word_to_idx)
        target = torch.tensor([[float(label)]])
        
        prediction = model(sentence_in)
        loss = criterion(prediction, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1} | Avg Loss: {total_loss/len(data):.4f}")

# --- STEP 4: OUTPUT / INFERENCE ---
def predict(sentence):
    model.eval()
    with torch.no_grad():
        seq = prepare_sequence(sentence, word_to_idx)
        prob = model(seq).item()
        sentiment = "POSITIVE" if prob > 0.5 else "NEGATIVE"
        print(f"Sentence: '{sentence}' | Sentiment: {sentiment} ({prob:.2f})")

print("\n--- RESULTS ---")
predict("I love LSTMs")
predict("the gradients are vanishing")