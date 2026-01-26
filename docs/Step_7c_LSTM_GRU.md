# Step 7c — LSTMs and GRUs (Advanced RNNs)

> **Goal:** Learn about LSTM and GRU, improvements over basic RNNs.  
> **Tools:** Python + PyTorch

---

## 7c.1 The Problem with Basic RNNs

- ❌ Vanishing gradient problem
- ❌ Can't remember long sequences
- ❌ Struggles with dependencies far apart

---

## 7c.2 What is LSTM?

**LSTM = Long Short-Term Memory**

Key innovation: **Gated architecture**
- ✅ Forget gate: What to forget
- ✅ Input gate: What to remember
- ✅ Output gate: What to output
- ✅ Cell state: Long-term memory

**Benefits:**
- Can remember information for long periods
- Solves vanishing gradient problem
- Better at long sequences

---

## 7c.3 What is GRU?

**GRU = Gated Recurrent Unit**

Simplified version of LSTM:
- ✅ Reset gate: How much past to forget
- ✅ Update gate: How much new info to add

**Benefits:**
- Simpler than LSTM (fewer parameters)
- Often performs similarly to LSTM
- Faster to train

---

## 7c.4 When to Use Each

**Basic RNN:**
- Simple tasks
- Short sequences
- Fast training

**LSTM:**
- Long sequences
- Complex dependencies
- When memory is critical

**GRU:**
- Good balance of performance and speed
- Often as good as LSTM
- Fewer parameters

---

## 7c.5 Applications

- Machine translation
- Chatbots
- Time series forecasting
- Music generation
- Text summarization
