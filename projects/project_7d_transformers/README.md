# Project 7d: Transformers and Attention

> **Build and understand Transformer architecture with attention mechanisms**

**Difficulty**: ⭐⭐⭐⭐ Expert  
**Time**: 4-5 hours  
**Prerequisites**: Steps 0-7 (Especially Step 7: RNNs, Step 7c: LSTM/GRU)

---

## 📚 Table of Contents

1. [Project Overview](#project-overview)
2. [Problem: Implement Attention and Transformer](#problem-implement-attention-and-transformer)
3. [Key Concepts](#key-concepts)
4. [Step-by-Step Guide](#step-by-step-guide)
5. [Expected Results](#expected-results)
6. [Extension Ideas](#extension-ideas)

---

## 🎯 Project Overview

This project teaches you to build Transformer models with attention mechanisms. You'll learn to:

- Implement self-attention from scratch
- Build Transformer encoder blocks
- Understand multi-head attention
- Apply Transformers to text tasks

### Why Transformers?

- **State-of-the-art**: Powers GPT, BERT, ChatGPT
- **Parallel processing**: Faster than RNNs
- **Better understanding**: Attention shows what model focuses on
- **Foundation for modern NLP**: Essential for advanced models

---

## 📋 Problem: Implement Attention and Transformer

### Task

Build a Transformer model for text classification:
1. **Implement Self-Attention**: Core mechanism
2. **Build Transformer Block**: Encoder architecture
3. **Multi-Head Attention**: Multiple attention heads
4. **Apply to Classification**: Text classification task

### Learning Objectives

- Understand attention mechanism
- Implement Transformer architecture
- Use positional encoding
- Apply to real tasks

---

## 🧠 Key Concepts

### 1. Attention Mechanism

**Purpose**: Focus on relevant parts of input

**Query, Key, Value**:
- Query: What we're looking for
- Key: What we're matching against
- Value: What we retrieve

**Formula**:
```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

### 2. Self-Attention

**Each position attends to all positions**: Learns relationships

**Benefits**:
- Parallel computation
- Long-range dependencies
- Interpretable (attention weights)

### 3. Multi-Head Attention

**Multiple attention heads**: Different learned relationships

**Process**:
1. Split into multiple heads
2. Apply attention to each head
3. Concatenate results

---

## 🚀 Step-by-Step Guide

### Step 1: Implement Self-Attention

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, d_model, d_k=None, d_v=None):
        super(SelfAttention, self).__init__()
        if d_k is None:
            d_k = d_model
        if d_v is None:
            d_v = d_model
        
        self.d_k = d_k
        self.W_q = nn.Linear(d_model, d_k)
        self.W_k = nn.Linear(d_model, d_k)
        self.W_v = nn.Linear(d_model, d_v)
    
    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        batch_size, seq_len, d_model = x.size()
        
        # Compute Q, K, V
        Q = self.W_q(x)  # (batch, seq_len, d_k)
        K = self.W_k(x)  # (batch, seq_len, d_k)
        V = self.W_v(x)  # (batch, seq_len, d_v)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
```

### Step 2: Multi-Head Attention

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        
        # Linear projections
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        # Output projection
        output = self.W_o(attention_output)
        
        return output, attention_weights
```

### Step 3: Positional Encoding

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x
```

### Step 4: Transformer Block

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Self-attention with residual
        attn_output, _ = self.attention(x)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x
```

### Step 5: Complete Transformer Model

```python
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_heads=4, num_layers=2, 
                 d_ff=512, num_classes=2, max_len=100, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.classifier = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Embedding + positional encoding
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Classification (use first token or mean pooling)
        x = x.mean(dim=1)  # Mean pooling
        output = self.classifier(x)
        
        return output
```

---

## 📊 Expected Results

### Training Output

```
Training Transformer Model...
Epoch 10/100, Loss: 0.5234
Epoch 20/100, Loss: 0.3456
...
Epoch 100/100, Loss: 0.1234

Test Accuracy: 89.5%
```

### Attention Visualization

- Attention weights show which words the model focuses on
- Higher attention = more important for classification

---

## 💡 Extension Ideas

1. **BERT-style Pre-training**
   - Masked language modeling
   - Next sentence prediction
   - Fine-tuning for tasks

2. **GPT-style Generation**
   - Causal attention mask
   - Autoregressive generation
   - Text completion

3. **Vision Transformers**
   - Apply to images
   - Patch embeddings
   - Image classification

---

## ✅ Success Criteria

- ✅ Implement self-attention correctly
- ✅ Build Transformer architecture
- ✅ Use positional encoding
- ✅ Achieve good classification performance
- ✅ Visualize attention weights

---

**Ready to build Transformers? Let's implement attention!** 🚀
