# 📚 Day 1 Notes: RNN Fundamentals & Time-Series Comparisons

## 1. How an RNN Works Internally

Unlike classical neural networks that process inputs independently, Recurrent Neural Networks (RNNs) process data sequentially. They contain a "loop" where the output of one step becomes part of the input for the next.

* **The Hidden State (Memory):** At each time step $t$, the RNN combines the current input ($x_t$) with the "hidden state" from the previous step ($h_{t-1}$) to produce a new hidden state ($h_t$).
* **The Core Math:** $h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$
* $W_{xh}$: Weights for the current input.
* $W_{hh}$: Weights for the previous hidden state.
* $\tanh$: The activation function that squeezes values between -1 and 1 to maintain stability.



## 2. The Vanishing Gradient Problem (Short-Term Memory)

Standard RNNs suffer from a "short-term memory" issue, meaning they struggle to carry information from the beginning of a long sequence (like a long sentence or 100 days of stock data) to the end.

* **Why it happens:** During backpropagation, gradients are multiplied by the shared weight matrix ($W_{hh}$) at every time step. If these weights are less than 1, the gradient shrinks exponentially as it moves backward.
* **The Result:** The signal eventually reaches near-zero ("vanishes"), causing the network to stop updating the weights for early inputs. (This is why modern NLP uses LSTMs and Transformers).

## 3. RNNs vs. Classical Machine Learning (Linear Regression)

We compared how an RNN handles sequential data versus a classical model like Linear Regression.

| Feature | RNN (Sequence-Aware) | Linear Regression (Classical ML) |
| --- | --- | --- |
| **Data Handling** | Processes data step-by-step; maintains an evolving "memory." | Processes all inputs simultaneously as independent features. |
| **Order Sensitivity** | **High.** Shuffling the data changes the output entirely. | **Low.** Assigns fixed weights to fixed "slots." |
| **Input Flexibility** | Can dynamically handle sequences of 5 days or 50 days. | Requires a fixed input size (e.g., exactly 9 days). |

**Key Takeaway for NLP:** Linear Regression (or Bag-of-Words) sees the words "not" and "good" independently. An RNN processes "not", remembers it, and flips the meaning when it sequentially hits "good".

## 4. Time-Series Forecasting: ARIMA vs. RNN

We tested both models on a 30-day synthetic stock market dataset (Random Walk with upward drift).

* **ARIMA (AutoRegressive Integrated Moving Average):**
* *Approach:* Statistical. It simplifies the series to make it stationary and relies on recent lags and errors.
* *Strengths:* Fast, doesn't require a GPU, and excellent for short-term, stable trends. Gives a "safe" mathematical prediction.


* **RNN (Deep Learning):**
* *Approach:* Sequential memory loop.
* *Strengths:* Capable of capturing highly complex, non-linear patterns.
* *Weaknesses:* With only 30 days of data, an RNN is prone to overfitting (memorizing the noise instead of the actual trend).



## 5. Implementation & Code Highlights

* **PyTorch Mechanics:** We used `torch.nn.RNN(input_size=1, hidden_size=4, batch_first=True)` to create our model.
* **Data Reshaping:** Neural networks require data scaling (MinMaxScaler) and reshaping into 3D tensors: `[Batch_Size, Sequence_Length, Input_Dim]`.
* **Random Initialization:** We observed that an untrained RNN and Linear Regression output completely different random numbers due to the RNN passing data through a $\tanh$ loop multiple times versus a single weighted sum.
