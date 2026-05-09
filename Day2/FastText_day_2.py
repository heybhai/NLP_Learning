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

# 2. Train Word2Vec (Skip-gram)
# vector_size=100 is standard, min_count=1 to ensure all words are kept
w2v_model = Word2Vec(sentences=corpus, vector_size=100, window=5, min_count=1, workers=4, sg=1)

# 3. Train FastText
# Note: FastText learns subwords (character n-grams)
ft_model = FastText(sentences=corpus, vector_size=100, window=5, min_count=1, workers=4)

# 4. Load Pre-trained GloVe (via Gensim Downloader)
# This uses the 50-dimensional Wikipedia vectors
print("Loading pre-trained GloVe model...")
glove_model = api.load("glove-wiki-gigaword-50")

# --- COMPARISON TEST: The OOV Challenge ---

# We will test a word NOT in the training corpus: "scientists" (plural)
test_word = "scientists"

def check_oov(model, name, word):
    try:
        if hasattr(model, 'wv'):
            vec = model.wv[word]
        else:
            vec = model[word]
        print(f"✅ {name}: Found vector for '{word}'")
    except KeyError:
        print(f"❌ {name}: '{word}' is Out-of-Vocabulary (OOV)")

print(f"\nComparing OOV handling for '{test_word}':")
print("-" * 40)
check_oov(w2v_model, "Word2Vec", test_word)
check_oov(glove_model, "GloVe", test_word)
check_oov(ft_model, "FastText", test_word)

# --- ANALOGY TEST: Mathematical Context ---
# king - man + woman = queen
print("\nTesting Analogy (GloVe): King - Man + Woman")
result = glove_model.most_similar(positive=['king', 'woman'], negative=['man'], topn=1)
print(f"Result: {result}")

