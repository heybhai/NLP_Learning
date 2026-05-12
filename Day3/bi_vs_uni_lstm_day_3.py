import torch
import torch.nn as nn

# 1. Setup Mock Data
corpus = ["the", "apple", "company", "announced", "a", "new", "phone"]
word2idx = {word: idx for idx, word in enumerate(corpus)}
vocab_size = len(word2idx)

embedding_dim = 4
hidden_dim = 8

# Convert sentence to tensor: "the apple company announced a new phone"
sentence = [word2idx[w] for w in corpus]
# Shape: [Batch Size, Sequence Length]
inputs = torch.tensor([sentence]) 

# 2. Define Both Models
embedding = nn.Embedding(vocab_size, embedding_dim)

# Normal LSTM: reads left to right only
normal_lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=False)

# Bi-LSTM: reads both ways
bi_lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)

# 3. Process the Sentence
embedded = embedding(inputs)

# outputs contain the hidden states for EVERY word in the sentence
normal_outputs, _ = normal_lstm(embedded)
bi_outputs, _ = bi_lstm(embedded)

# 4. Analyze the word "apple" (Index 1)
print(f"Sentence: {' '.join(corpus)}")
print(f"Target Word: '{corpus[1]}' (Index 1)\n")
print("-" * 50)

# --- NORMAL LSTM ANALYSIS ---
normal_apple_state = normal_outputs[0, 1, :] # [Batch 0, Word Index 1, Features]
print("1. Normal LSTM Output for 'apple'")
print(f"Shape: {normal_apple_state.shape} (Matches hidden_dim of {hidden_dim})")
print("Context captured: ['the', 'apple']")
print("Missing context: ['company', 'announced', 'a', 'new', 'phone']")
print("-" * 50)

# --- BI-LSTM ANALYSIS ---
bi_apple_state = bi_outputs[0, 1, :] # [Batch 0, Word Index 1, Features]
print("2. Bi-LSTM Output for 'apple'")
print(f"Shape: {bi_apple_state.shape} (Matches hidden_dim * 2: {hidden_dim * 2})")

# The Bi-LSTM output is a concatenation of the Forward and Backward hidden states.
# We can literally split it in half to see what it knows.
forward_half = bi_apple_state[:hidden_dim]
backward_half = bi_apple_state[hidden_dim:]

print(f"\n-> Forward Half Shape: {forward_half.shape}")
print("   Context captured: ['the', 'apple']")

print(f"-> Backward Half Shape: {backward_half.shape}")
print("   Context captured: ['phone', 'new', 'a', 'announced', 'company', 'apple']")

print("\nConclusion: The Bi-LSTM representation for 'apple' contains the full ")
print("sentence context, allowing it to easily classify 'apple' as an organization ")
print("rather than a fruit, whereas the Normal LSTM has to guess blindly at this time step.")