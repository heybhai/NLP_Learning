The transition from standard RNNs to LSTMs (Long Short-Term Memory) is a transition from a system that only has "short-term memory" to one that can maintain a "long-term state."In a vanilla RNN, the hidden state is constantly being overwritten by a $\tanh$ function at every step. In an LSTM, we introduce a Cell State ($C_t$), which acts like a "conveyor belt" or "high-speed rail" that runs through the entire sequence with only minor linear interactions, allowing information to flow for hundreds of time steps without vanishing.

The Internal Anatomy of an LSTM
The behavior of the Cell State is controlled by three specialized "gates." Each gate is composed of a sigmoid neural net layer and a pointwise multiplication operation.
1. The Forget Gate ($f_t$)Goal: 
    Decide what information to discard from the cell state.How it works: It looks at the previous hidden state ($h_{t-1}$) and the current input ($x_t$), and outputs a number between 0 (completely forget) and 1 (completely keep) for each number in the cell state $C_{t-1}$.Example: If you are processing a news article and the subject changes from "Data Science" to "Motorcycles," the forget gate clears the "Data Science" context from memory.
2. The Input Gate ($i_t$ & $\tilde{C}_t$)Goal: 
    Decide what new information to store in the cell state.How it works:A sigmoid layer (the "input gate layer") decides which values will be updated.A $\tanh$ layer creates a vector of new candidate values, $\tilde{C}_t$, that could be added to the state.Math: These two are multiplied to determine what specifically is "worth" adding to the long-term memory.
3. The Output Gate ($o_t$)Goal: 
    Decide what the "working memory" (hidden state) should be for the next step.How it works: This is a filtered version of the cell state. We pass the cell state through a $\tanh$ (to push values between -1 and 1) and multiply it by the output of a sigmoid gate to decide which parts of the internal memory are relevant right now.

Feature,ARIMA,Vanilla RNN,LSTM
Logic,Statistical (Autoregressive),Neural (Hidden State),Neural (Cell State + Gates)
Memory,"Linear lags (Fixed p,d,q)",Short-term (Overwritten),Long-term (Gated flow)
Complexity,Low,Medium,High
Vanishing Gradient,N/A,High Risk,Low Risk
Best Use Case,"Stationary, linear data.",Very short sequences.,"Complex sequences, NLP, Time-series."

The Mathematical Update Equations
For your technical documentation, the core of the LSTM "Magic" is how it updates the cell state ($C_t$):$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$$$C_t = f_t * C_{t-1} + i_t * \tilde{C}_t$$
Why does this prevent vanishing gradients?
In a Vanilla RNN, the gradient is multiplied by the weight matrix $W$ at every step (leading to $W^n$).In an LSTM, the gradient flows through the cell state $C_t$ via addition (the $+$ in the last equation). Addition distributes the gradient equally, preventing it from shrinking to zero as it moves back through time.

1. The "Constant Error Carousel"We will study the mathematical heart of the LSTM—the Cell State ($C_t$). Think of it as a high-speed rail line that runs through the entire sequence. Information can stay on this rail unchanged for a long time, allowing the gradient to flow backward without vanishing.
2. The Gating Mechanisms (The "Three Guards")LSTMs use three "gates" to decide what information enters, stays, or leaves the cell state. We will dive into the math of each:The Forget Gate ($f_t$): Decides what piece of the past is no longer relevant (e.g., a subject change in a sentence).The Input Gate ($i_t$): Decides what new information from the current word is worth storing.The Output Gate ($o_t$): Decides which part of the internal memory should be revealed as the hidden state for the next step.
3. The Cell State vs. Hidden StateWe will clarify a common point of confusion: the difference between Long-term memory ($C_t$) and Short-term memory/Working memory ($h_t$).
4. Mathematical ImplementationWe will look at the update equations that make this possible. Specifically, why using addition for the cell state update (instead of the repeated multiplication used in Vanilla RNNs) is the "magic trick" that prevents gradients from vanishing.$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$