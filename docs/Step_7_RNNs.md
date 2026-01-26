# Step 7 — RNNs (Recurrent Neural Networks for Sequences)

> **Goal:** Learn how to process sequences (text, time series) using Recurrent Neural Networks.  
> **Tools:** Python + PyTorch + NumPy

---

## 7.1 Big Idea: Why RNNs?

Regular neural networks:
- Process **one input → one output**
- No memory of previous inputs
- Can't handle sequences

**RNNs (Recurrent Neural Networks):**
- Process **sequences** (text, time series, music)
- **Remember** previous information
- Perfect for: language, stock prices, speech, etc.

🧠 **Key insight:**  
RNNs have a "memory" that helps them understand context.

---

## 7.2 The Problem with Regular Networks

Imagine predicting the next word in a sentence:

> "The cat sat on the..."

A regular network sees only the last word: "the"  
An RNN sees: "The cat sat on the" and remembers the context.

---

## 7.3 Simple Sequence Example

Let's start simple: **predict the next number**

Sequence: `[1, 2, 3, 4]` → next should be `5`

```python
# Create sequence data
sequences = [
    [1, 2, 3, 4],  # → 5
    [2, 3, 4, 5],  # → 6
    [5, 6, 7, 8],  # → 9
]
```

---

## 7.4 How RNNs Work

An RNN processes sequences **step by step**:

```
Input: [1, 2, 3, 4]
        ↓
Step 1: Process 1 → hidden state
Step 2: Process 2 → hidden state (remembers 1)
Step 3: Process 3 → hidden state (remembers 1, 2)
Step 4: Process 4 → hidden state (remembers 1, 2, 3)
        ↓
Output: Predict 5
```

The **hidden state** is the "memory" of the RNN.

---

## 7.5 Building an RNN in PyTorch

```python
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, output_size=1):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        rnn_out, hidden = self.rnn(x)
        last_output = rnn_out[:, -1, :]  # Use last step
        prediction = self.fc(last_output)
        return prediction
```

🧠 **Key components:**
- `nn.RNN`: The recurrent layer
- `hidden_size`: Size of the memory
- `batch_first=True`: Batch dimension first

---

## 7.6 Data Shape for RNNs

RNNs need data in a specific shape:

```
(batch_size, sequence_length, features)
```

Example:
- Batch of 10 sequences
- Each sequence has 5 steps
- Each step has 1 feature

Shape: `(10, 5, 1)`

---

## 7.7 Training an RNN

```python
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(500):
    predictions = model(X)
    loss = loss_fn(predictions, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

Same training process as before, but now processing sequences!

---

## 7.8 Text Sequences (Character-level)

RNNs can process text **character by character**:

```python
text = "hello"
# Convert to numbers
char_to_idx = {'h': 0, 'e': 1, 'l': 2, 'o': 3}

# Create sequences
"hel" → predict 'l'
"ell" → predict 'o'
```

This is how text generators work!

---

## 7.9 Real-World Applications

### Language Translation
- Input: English sentence
- Output: Spanish sentence
- RNN processes word by word

### Text Generation
- Input: "Once upon a time"
- Output: "...there was a princess"
- RNN generates next words

### Speech Recognition
- Input: Audio waveform
- Output: Text transcription
- RNN processes time steps

### Stock Price Prediction
- Input: Past prices
- Output: Future price
- RNN remembers trends

---

## 7.10 RNN Architecture Deep Dive

### The Recurrent Cell

```
h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b)
```

Where:
- `h_t`: Current hidden state (memory)
- `h_{t-1}`: Previous hidden state
- `x_t`: Current input
- `W_hh`, `W_xh`: Weight matrices

🧠 **The magic:** The hidden state flows from step to step!

---

## 7.11 Limitations of Simple RNNs

❌ **Vanishing Gradient Problem**
- Can't remember very long sequences
- Gradients become too small

❌ **Short-term Memory**
- Struggles with long dependencies
- Forgets early information

---

## 7.12 Better Alternatives

### LSTM (Long Short-Term Memory)
- ✅ Better at remembering long sequences
- ✅ Has "gates" to control memory
- ✅ Used in many applications

### GRU (Gated Recurrent Unit)
- ✅ Simpler than LSTM
- ✅ Still good memory
- ✅ Faster training

### Transformer
- ✅ Uses attention mechanism
- ✅ No recurrence needed
- ✅ Powers GPT, BERT, etc.

---

## 7.13 Mini Exercises

### Exercise 1
Modify the sequence prediction:
- Try different patterns (multiply by 2, etc.)
- Change sequence length

### Exercise 2
Build a character-level text generator:
- Train on a longer text
- Generate new text

### Exercise 3
Experiment with hidden size:
- Try `hidden_size=16`
- Try `hidden_size=64`
- Observe the difference

---

## 7.14 Checklist (Before Moving On)

Students should understand:
- ✅ Why RNNs are needed for sequences
- ✅ How RNNs process data step by step
- ✅ How to build an RNN in PyTorch
- ✅ What the hidden state represents
- ✅ Basic text processing concepts

If YES → ready for advanced topics!

---

## 7.15 Next Step Preview

You've now learned:
- ✅ Feedforward networks (Steps 1-5)
- ✅ PyTorch framework (Step 6)
- ✅ Recurrent networks (Step 7)

**What's next?**
- CNNs (Convolutional Neural Networks) for images
- Advanced RNNs (LSTM, GRU)
- Transformers and attention
- Real-world projects

---

## 🎉 Congratulations!

You've completed Step 7: RNNs!

You now understand:
- How AI processes sequences
- How to build RNNs
- Applications in text and time series

**You're becoming a real AI engineer!** 🚀
