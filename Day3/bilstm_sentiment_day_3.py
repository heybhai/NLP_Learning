import torch
import torch.nn as nn
import torch.optim as optim

# 1. Define the Bi-LSTM Model
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(BiLSTMClassifier, self).__init__()
        
        # Word Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Bi-directional LSTM Layer
        self.lstm = nn.LSTM(embedding_dim, 
                            hidden_dim, 
                            num_layers=1, 
                            bidirectional=True, 
                            batch_first=True)
        
        # Fully Connected Layer
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text):
        embedded = self.embedding(text) 
        output, (hidden, cell) = self.lstm(embedded)
        
        # Concatenate the final forward and backward hidden states
        final_hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        logits = self.fc(final_hidden)
        return self.sigmoid(logits)

# 2. Setup Actual Vocabulary and Data
# Define the corpus and create a mapping from word to index
corpus = ["i", "love", "this", "hate", "it","apple","phone","fruit","<pad>", "<unk>"]
word2idx = {word: idx for idx, word in enumerate(corpus)}

vocab_size = len(word2idx) 
embedding_dim = 16
hidden_dim = 32
output_dim = 1 # Binary: 0 or 1

model = BiLSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)

# Define actual sentences
train_sentences = [
    ["i", "love", "this"],
    ["i", "hate", "it"],
    ["i", "love", "apple"],
    ["i", "hate", "phone"],
    ["i", "love", "fruit"],
    ["i", "hate", "fruit"]
]

# Convert sentences to integer tensors using the vocabulary
train_seqs = [[word2idx[w] for w in sent] for sent in train_sentences]
inputs = torch.tensor(train_seqs) 
labels = torch.tensor([[1.0], [0.0], [1.0], [0.0], [1.0], [0.0]]) # 1=Positive, 0=Negative

# 3. Training Setup
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 4. Simple Training Loop
print("Starting Training...")
for epoch in range(20):
    model.train()
    optimizer.zero_grad()
    
    predictions = model(inputs)
    loss = criterion(predictions, labels)
    
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

# 5. Inference
model.eval()
with torch.no_grad():
    # Define an actual test sentence
    raw_test_sentence = ["i", "love", "it"]
    
    # Map words to indices, using <unk> if the word isn't in our vocab
    test_seq = [word2idx.get(w, word2idx["<unk>"]) for w in raw_test_sentence]
    
    # Needs a batch dimension, so we wrap it in an extra list
    test_tensor = torch.tensor([test_seq]) 
    
    prob = model(test_tensor).item()
    sentiment = "Positive" if prob > 0.5 else "Negative"
    
    print(f"\nTest Sentence: '{' '.join(raw_test_sentence)}'")
    print(f"Test Result: '{sentiment}' (Confidence: {prob:.4f})")