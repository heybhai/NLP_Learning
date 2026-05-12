# 🚀 90 Days of NLP: From Zero to Hero

Welcome to my personal learning log and portfolio for my 90-day intensive Natural Language Processing (NLP) journey! 

**Goal:** To transition from basic text processing to mastering state-of-the-art Large Language Models (LLMs) and sequential architectures, dedicating 2 hours a day to theory and practical implementation.

---

## 🗺️ The 90-Day Roadmap

| Phase | Focus | Key Topics |
| :--- | :--- | :--- |
| **Phase 1: Foundations** | Processing & Embeddings | Tokenization, Word2Vec, GloVe, FastText, Doc2Vec |
| **Phase 2: Sequences** | Deep Learning Models | RNNs, LSTMs, GRUs, Bi-LSTMs, Seq2Seq |
| **Phase 3: Attention** | Transformers | Attention Mechanism, BERT, GPT Architecture |
| **Phase 4: Advanced** | LLMs & Applications | Fine-tuning, RAG, Prompt Engineering, Deployment |

---

## 📅 Daily Learning Log

### ✅ Week 1: Embeddings & Sequential Memory
*(Currently In Progress)*

* **Day 1 & Day 2: Advanced Word Embeddings**
    * **Topics:** Word2Vec (Skip-gram/CBOW), GloVe (Global Co-occurrence), and FastText (Subword n-grams).
    * **Key Concept:** Moving from sparse One-Hot Encoding to dense vector spaces. Understanding how FastText solves the Out-of-Vocabulary (OOV) problem.
    * **Implementation:** Trained Word2Vec and FastText models using `gensim` and tested OOV handling.

* **Day 3: Solving the Memory Bottleneck**
    * **Topics:** Vanilla RNNs, LSTMs, Bi-LSTMs, and GRUs.
    * **Key Concept:** The Vanishing Gradient problem and how Gated Architectures (Forget, Input, Output, Reset, Update) solve it by using a long-term cell state.
    * **Implementation:** Built PyTorch sequence classifiers and compared ARIMA, Vanilla RNN, and LSTM for time-series forecasting (Fractal Analytics stock data).

* **Day 4 & 5: Sequence-to-Sequence (Seq2Seq)**
    * **Topics:** Encoder-Decoder architectures, handling variable-length inputs and outputs (e.g., Machine Translation).
    * **Focus:** Understanding the bottleneck of passing a single "Context Vector" between the Encoder and Decoder.

* **Day 6 & 7: Introduction to Attention**
    * **Topics:** Bahdanau and Luong Attention. 
    * **Focus:** How allowing the Decoder to "look back" at specific parts of the Encoder sequence fixes the Seq2Seq bottleneck.

* **Day 8, 9 & 10: The Transformer Revolution**
    * **Topics:** Deep dive into the *"Attention is All You Need"* paper.
    * **Focus:** Stripping away recurrence entirely. Self-Attention, Multi-Head Attention, and Positional Encoding.

*(More weeks to be added as the journey progresses...)*

---

## 🛠️ Tech Stack
* **Languages:** Python
* **Libraries/Frameworks:** PyTorch, Gensim, Scikit-Learn, Pandas, NumPy, Statsmodels
* **Tools:** Jupyter Notebooks, Git