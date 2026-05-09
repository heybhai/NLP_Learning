import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 1. Our "Toy" Corpus
corpus = [
    "ice is solid", "ice is cold", "steam is gas", "steam is hot", 
    "water is liquid", "water is ice", "water is steam", "ice and steam are different", "water is in between ice and steam"
]

# 2. Build a Vocabulary and Co-occurrence Matrix
words = sorted(list(set(" ".join(corpus).split())))
word2idx = {w: i for i, w in enumerate(words)}
X = np.zeros((len(words), len(words)))

window_size = 1
for sentence in corpus:
    tokens = sentence.split()
    for i, token in enumerate(tokens):
        # Check context words in window
        start = max(0, i - window_size)
        end = min(len(tokens), i + window_size + 1)
        for j in range(start, end):
            if i != j:
                X[word2idx[token], word2idx[tokens[j]]] += 1

# 3. Simulate "GloVe-like" Vectors
# In real GloVe, we use SGD. Here, we use SVD as a shortcut for visualization.
# We take the Log of X (handling zeros) to match GloVe's objective
X_log = np.log(X + 1) 
u, s, vh = np.linalg.svd(X_log)
vectors = u[:, :2] # Reduce to 2 dimensions for plotting

# 4. Visualization
plt.figure(figsize=(10, 7))
plt.scatter(vectors[:, 0], vectors[:, 1], color='red')

for i, word in enumerate(words):
    plt.annotate(word, (vectors[i, 0], vectors[i, 1]), fontsize=12)

plt.title("Visualizing Global Co-occurrence (GloVe Logic)")
plt.grid(True)
plt.show()