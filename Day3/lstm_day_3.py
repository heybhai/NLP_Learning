import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
from gensim.models import Word2Vec, FastText
import gensim.downloader as api
import numpy as np

# 1. Prepare a small real-world-style corpus
# In a production environment, this would be your cleaned dataframe column
corpus = [
    ["the", "senior", "data", "scientist", "is", "coding"],
    ["machine", "learning", "is", "a", "subset", "of", "ai"],
    ["nlp", "pipelines", "require", "word", "embeddings"],
    ["fasttext", "handles", "out", "of", "vocabulary", "words"],
    ["word2vec", "treats", "each", "word", "as", "an", "atomic", "unit"]
]

ft_model = FastText(sentences=corpus, vector_size=100, window=5, min_count=1, workers=4)
# 1. Prepare the Weights from Day 2 (Assuming ft_model is your FastText model)
# We create a weight matrix where each row is the vector for a vocab index.
def create_embedding_matrix(gensim_model):
    vocab = gensim_model.wv.index_to_key
    vector_size = gensim_model.vector_size
    # Matrix size: (VocabSize + 1) x VectorDim (the +1 is for padding)
    weights_matrix = np.zeros((len(vocab) + 1, vector_size))
    
    for i, word in enumerate(vocab):
        weights_matrix[i] = gensim_model.wv[word]
        
    return torch.tensor(weights_matrix, dtype=torch.float32)

# 2. Define the LSTM Classifier
class LSTMTextClassifier(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, output_dim):
        super(LSTMTextClassifier, self).__init__()
        
        # Initialize Embedding layer with pre-trained weights
        num_embeddings, embedding_dim = embedding_matrix.size()
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)
        
        # LSTM Layer
        # batch_first=True means input shape is (batch, seq_len, features)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=False)
        
        # Output Layer (Linear)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x) # (batch, seq_len, embed_dim)
        
        # lstm_out: (batch, seq_len, hidden_dim)
        # hidden_state (h_n): (1, batch, hidden_dim)
        lstm_out, (h_n, c_n) = self.lstm(embedded)
        
        # We only care about the last hidden state of the sequence
        final_feature_map = h_n[-1] 
        
        logits = self.fc(final_feature_map)
        return logits

# 3. Instantiate
# Assuming output_dim=2 for Sentiment Analysis (Positive/Negative)
# Assuming ft_model from yesterday's session is in memory
weights = create_embedding_matrix(ft_model)
model = LSTMTextClassifier(embedding_matrix=weights, hidden_dim=64, output_dim=2)

print(model)

class TextDataset(Dataset):
    def __init__(self, texts, labels, word2idx):
        self.labels = labels
        # Convert words to indices using the vocab from Day 2
        self.sequences = [
            torch.tensor([word2idx.get(w, 0) for w in text], dtype=torch.long)
            for text in texts
        ]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], torch.tensor(self.labels[idx], dtype=torch.long)

def collate_fn(batch):
    """Handles padding for variable-length sentences in a batch."""
    sequences, labels = zip(*batch)
    # Pad sequences to the length of the longest sentence in the batch
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    return padded_sequences, torch.stack(labels)
def train_model(model, train_loader, val_loader, epochs=5, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 1. Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 2. Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Simple Validation Check
        val_acc = evaluate(model, val_loader, device)
        print(f"Loss: {train_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}%")

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    return 100 * correct / total
# 1. Setup Vocab and Data
# vocab maps word -> index based on your FastText index_to_key
word2idx = {word: i for i, word in enumerate(ft_model.wv.index_to_key)}

train_texts = [["the", "code", "is", "great"], ["this", "pipeline", "failed"]]
train_labels = [1, 0] # 1: Positive, 0: Negative

dataset = TextDataset(train_texts, train_labels, word2idx)
loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

# 2. Initialize the Model (from previous turn)
# weights = pre-trained matrix from FastText
model = LSTMTextClassifier(embedding_matrix=weights, hidden_dim=128, output_dim=2)

# 3. Execute Training
train_model(model, loader, loader, epochs=3)

'''
The Architecture: LSTM Sequence 
ClassifierThe data flow works as follows:
    Input: A sequence of token IDs.
    Embedding Layer: Maps IDs to the 100D vectors we discussed yesterday.
    LSTM Layer: Processes vectors one by one, maintaining the Cell State ($C_t$) for long-term memory.
    Fully Connected Layer: Takes the final hidden state ($h_n$) and maps it to your class probabilities.

Training Loops and Loss Calculation
This is where we implement the forward pass, loss calculation, and backpropagation. 
For a classification task, we use Cross Entropy Loss.
$$L = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \log(\hat{y}_{i,c})$$
'''