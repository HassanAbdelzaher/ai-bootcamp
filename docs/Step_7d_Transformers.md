# Step 7d — Transformers (BERT, GPT)

> **Goal:** Introduction to Transformer architecture and attention mechanism.  
> **Tools:** Python + PyTorch

---

## 7d.1 The Evolution

**RNN (1980s):**
- Sequential processing
- Vanishing gradients

**LSTM (1997):**
- Solved vanishing gradients
- Better memory
- Still sequential

**Transformer (2017):**
- Parallel processing
- Attention mechanism
- No recurrence needed!
- Powers GPT, BERT, ChatGPT

---

## 7d.2 What is Attention?

**Attention = Focus on relevant parts**

Example: "The cat sat on the mat"
- When processing "mat", attention focuses on "cat", "sat", "on"

**Self-Attention:**
- Each word attends to all other words
- Learns relationships
- Parallel computation

---

## 7d.3 Transformer Architecture

**Components:**
1. Encoder (for understanding)
2. Decoder (for generation)
3. Multi-Head Attention
4. Position Encoding
5. Feed-Forward Networks

---

## 7d.4 BERT Overview

**BERT = Bidirectional Encoder Representations from Transformers**

**Key features:**
- ✅ Bidirectional: Reads left-to-right AND right-to-left
- ✅ Pre-trained on massive text corpus
- ✅ Fine-tuned for specific tasks

**Applications:**
- Text classification
- Question answering
- Named entity recognition
- Sentiment analysis

---

## 7d.5 GPT Overview

**GPT = Generative Pre-trained Transformer**

**Key features:**
- ✅ Autoregressive: Generates text one token at a time
- ✅ Unidirectional: Reads left-to-right
- ✅ Pre-trained then fine-tuned

**Applications:**
- Text generation
- Chatbots
- Code generation
- Creative writing

---

## 7d.6 BERT vs GPT

**BERT:**
- Better for understanding tasks
- Sees full context (bidirectional)
- Great for classification
- Not good for generation

**GPT:**
- Better for generation tasks
- Autoregressive generation
- Great for creative tasks
- Only sees left context

---

## 7d.7 Why Transformers are Revolutionary

✅ **Parallel processing:**
- Process entire sequence at once
- Much faster than RNNs

✅ **Long-range dependencies:**
- Attention connects any two positions
- No information loss over distance

✅ **Scalability:**
- Can handle very long sequences
- Powers large language models

✅ **Transfer learning:**
- Pre-train on large corpus
- Fine-tune for specific tasks

---

## 7d.8 Real-World Impact

Transformers power:
- 🤖 ChatGPT (GPT-4)
- 🔍 Google Search (BERT)
- 🌐 GitHub Copilot
- 🎨 DALL-E
- 🗣️ Voice assistants
- 📝 Translation services
