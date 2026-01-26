# Step 7a — Text Generator (Character-level RNN)

> **Goal:** Build a character-level text generator using RNNs.  
> **Tools:** Python + PyTorch

---

## 7a.1 Overview

Learn to generate text character by character using an RNN. This is how early text generators worked!

---

## 7a.2 Key Concepts

- **Character-level modeling**: Predict next character given previous characters
- **Embedding layer**: Converts characters to vectors
- **Temperature sampling**: Controls randomness in generation
- **Gradient clipping**: Prevents exploding gradients

---

## 7a.3 How It Works

1. Train on text corpus
2. Learn patterns in character sequences
3. Generate new text by sampling next characters
4. Use temperature to control creativity

---

## 7a.4 Applications

- Creative writing
- Code generation
- Chatbots
- Poetry generation

---

## 7a.5 Improvements

- Use word-level instead of character-level
- Train on larger datasets
- Use LSTM or GRU
- Try Transformer models (GPT)
