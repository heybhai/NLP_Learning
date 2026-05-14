
### 1. The Mathematics & Symbols Explained

Word2Vec (specifically the Skip-gram architecture described here) is trying to maximize the probability of seeing the correct context words given a central word.

Here is the translation of the symbols from the text:

* $t$: The current position (index) in the text corpus.
* $m$: The context window size (e.g., if $m=2$, you look 2 words to the left and 2 to the right).
* $w_t$: The central word at position $t$.
* $w_{t+j}$: A context word within the window.
* $\theta$: The parameters of our model. In Word2Vec, $\theta$ is simply the two massive embedding matrices (the $v$ vectors for center words and $u$ vectors for context words).

#### The Objective Function

The model wants to minimize the average negative log-likelihood:


$$J(\theta) = -\frac{1}{T} \sum_{t=1}^T \sum_{-m \le j \le m, j \ne 0} \log P(w_{t+j} | w_t ; \theta)$$

* **Why log?** Probabilities are tiny decimals. Multiplying millions of them causes computers to underflow to zero. Taking the log allows us to add them instead.
* **Why negative?** In machine learning, we typically *minimize* loss functions. Maximizing probability is mathematically identical to minimizing negative log-likelihood.

#### The Softmax Function

To calculate the probability $P(o|c)$ of an outside context word ($o$) given a central word ($c$), we use Softmax:


$$P(o|c) = \frac{\exp(u_o^\top v_c)}{\sum_{w \in V} \exp(u_w^\top v_c)}$$

* **The Numerator (The Target):** $u_o^\top v_c$ is the dot product (similarity) between the central word's vector and the actual context word's vector. We take the exponent ($\exp$) to ensure the result is always positive.
* **The Denominator (The Normalization):** We calculate the dot product between the central word $v_c$ and **every single word in the vocabulary** $u_w$. Summing all these up ensures our final output is a valid probability distribution (sums to 1).

**The Gradient Descent Paradox:**
The text notes that minimizing this loss forces the model to decrease the similarity between "cat" and *all* other words in the vocabulary—even words like "grey" which might actually be good context words!

* **Why it works:** Don't worry. Right now, at step $t$, the word is "cute". The model pushes "cat" and "cute" closer, and pushes "cat" and "grey" apart. But a few sentences later, the corpus might say "the grey cat". At that step, the model pushes "cat" and "grey" closer together. Over millions of updates, the vectors settle into an equilibrium that represents their true average distribution in the language.

---

### 2. How does Word2Vec choose the central word? (Does it have to be "cat"?)

It does **not** choose "cat" because "cat" is special. It doesn't choose at all.

Word2Vec is a brute-force algorithm. It iterates through the entire text corpus sequentially, shifting one word at a time from the very first word of the document to the very last.

* If your sentence is: "The cute cat is playing."
* **Step 1:** Center word is "The". Context is "cute", "cat".
* **Step 2:** Center word is "cute". Context is "The", "cat", "is".
* **Step 3:** Center word is "cat". Context is "The", "cute", "is", "playing".

"Cat" was simply the word at position $t$ in Lena's specific textbook example.

---

### 3. Hyperparameters: How a Data Scientist tunes Word2Vec

When you are actually training these embeddings (using a library like `gensim`), you don't manually calculate the gradients. Instead, you set the hyperparameters.

#### Choosing the Window Size ($m$)

The window size determines what *kind* of similarity your embeddings will learn.

* **Small Window ($m = 2$ to $5$):** The model only sees words immediately next to each other. The embeddings will capture **syntactic** and functional similarity. (e.g., "cat" and "dog" will have similar vectors because they both immediately follow adjectives like "cute" or verbs like "pet").
* **Large Window ($m = 5$ to $15+$):** The model sees words far apart in the same sentence/paragraph. The embeddings will capture broader **topical** or semantic similarity. (e.g., "cat" might end up close to "veterinarian" or "meow", even if they rarely sit right next to each other in a sentence).

#### Choosing Gradient Descent Parameters

* **Learning Rate ($\alpha$):** Usually starts around $0.025$ and linearly decays to near zero as training progresses. This allows the vectors to make big jumps early on and fine-tune as they reach optimal positions.
* **Epochs:** How many times the algorithm passes over the entire corpus. Usually 5 to 30 epochs, depending on the size of your data.

*(Note on the math: Calculating that Softmax denominator over a 50,000-word vocabulary for every single step is computationally devastating. This is why we use **Negative Sampling**, which you'll implement in your code days! Instead of calculating the whole denominator, we just pick 5 to 10 random "noise" words to push away from.)*

