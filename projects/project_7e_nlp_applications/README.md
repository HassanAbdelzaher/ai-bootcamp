# Project 7e: NLP Applications

> **Build practical NLP systems: tokenization, embeddings, NER, sentiment analysis**

**Difficulty**: ⭐⭐⭐ Advanced  
**Time**: 4-5 hours  
**Prerequisites**: Steps 0-7 (Especially Step 7: RNNs, Step 7a: Text Generator)

---

## 📚 Table of Contents

1. [Project Overview](#project-overview)
2. [Problem: Multiple NLP Tasks](#problem-multiple-nlp-tasks)
3. [Key Concepts](#key-concepts)
4. [Step-by-Step Guide](#step-by-step-guide)
5. [Expected Results](#expected-results)
6. [Extension Ideas](#extension-ideas)

---

## 🎯 Project Overview

This project teaches you to build practical NLP applications. You'll implement:

1. **Tokenization**: Character, word, subword
2. **Word Embeddings**: One-hot, learned, pre-trained
3. **Named Entity Recognition (NER)**: Find people, places, organizations
4. **Sentiment Analysis**: Classify text sentiment
5. **Text Classification Pipeline**: End-to-end system

### Why NLP Applications?

- **Real-world impact**: Used in chatbots, search, translation
- **Practical skills**: Directly applicable to jobs
- **Foundation**: Prepares for advanced NLP
- **Portfolio**: Build impressive projects

---

## 📋 Problem: Multiple NLP Tasks

### Task 1: Tokenization

Implement different tokenization methods:
- Character-level
- Word-level
- Subword-level (BPE concept)

### Task 2: Word Embeddings

Create and use embeddings:
- One-hot encoding
- Learned embeddings
- Pre-trained embeddings (GloVe concept)

### Task 3: Named Entity Recognition

Build NER system to identify:
- Person names
- Locations
- Organizations

### Task 4: Sentiment Analysis

Classify text sentiment:
- Positive
- Negative
- Neutral

---

## 🧠 Key Concepts

### 1. Tokenization

**Character-level**: Each character is a token
**Word-level**: Each word is a token
**Subword-level**: Words split into pieces (handles unknown words)

### 2. Word Embeddings

**One-hot**: Sparse, high-dimensional
**Learned**: Dense, low-dimensional, task-specific
**Pre-trained**: General-purpose, transfer learning

### 3. Named Entity Recognition

**Sequence labeling**: Tag each word with entity type
**BIO tagging**: B-begin, I-inside, O-outside

### 4. Sentiment Analysis

**Classification task**: Text → Sentiment class
**Features**: Word counts, embeddings, sequences

---

## 🚀 Step-by-Step Guide

### Step 1: Tokenization

```python
import re
from collections import Counter

def character_tokenize(text):
    """Character-level tokenization"""
    return list(text.lower())

def word_tokenize(text):
    """Word-level tokenization"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return text.split()

def create_vocabulary(texts, min_freq=2):
    """Create vocabulary from texts"""
    word_counts = Counter()
    for text in texts:
        words = word_tokenize(text)
        word_counts.update(words)
    
    vocab = {'<PAD>': 0, '<UNK>': 1}
    idx = 2
    for word, count in word_counts.items():
        if count >= min_freq:
            vocab[word] = idx
            idx += 1
    
    return vocab

# Example
texts = [
    "I love machine learning",
    "Machine learning is fascinating",
    "I study AI and machine learning"
]

vocab = create_vocabulary(texts)
print(f"Vocabulary size: {len(vocab)}")
print(f"Vocabulary: {list(vocab.keys())[:10]}")
```

### Step 2: Word Embeddings

```python
import torch
import torch.nn as nn
import numpy as np

# One-hot encoding
def one_hot_encode(word, vocab_size):
    """One-hot encode a word"""
    one_hot = torch.zeros(vocab_size)
    one_hot[word] = 1.0
    return one_hot

# Learned embeddings
class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(EmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
    
    def forward(self, x):
        return self.embedding(x)

# Example
vocab_size = 1000
embed_dim = 128
embedding_layer = EmbeddingLayer(vocab_size, embed_dim)

# Get embedding for word
word_idx = 5
embedding = embedding_layer(torch.LongTensor([word_idx]))
print(f"Embedding shape: {embedding.shape}")  # (1, 128)
```

### Step 3: Named Entity Recognition

```python
class NERModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_tags=5):
        super(NERModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden_dim * 2, num_tags)  # *2 for bidirectional
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        output = self.classifier(lstm_out)
        return output

# NER tags: 0=PAD, 1=O, 2=B-PER, 3=I-PER, 4=B-LOC, 5=I-LOC, etc.
```

### Step 4: Sentiment Analysis

```python
class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_classes=3):
        super(SentimentClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, _) = self.lstm(embedded)
        # Use last hidden state
        output = self.classifier(hidden[-1])
        return output

# Classes: 0=Negative, 1=Neutral, 2=Positive
```

### Step 5: Complete Pipeline

```python
class NLPipeline:
    def __init__(self, vocab, model):
        self.vocab = vocab
        self.model = model
    
    def preprocess(self, text):
        """Preprocess text"""
        words = word_tokenize(text)
        indices = [self.vocab.get(word, self.vocab['<UNK>']) for word in words]
        return torch.LongTensor([indices])
    
    def predict(self, text):
        """Make prediction"""
        self.model.eval()
        with torch.no_grad():
            input_tensor = self.preprocess(text)
            output = self.model(input_tensor)
            prediction = torch.argmax(output, dim=1)
            return prediction.item()

# Usage
pipeline = NLPipeline(vocab, sentiment_model)
result = pipeline.predict("I love this product!")
print(f"Sentiment: {['Negative', 'Neutral', 'Positive'][result]}")
```

---

## 📊 Expected Results

### Tokenization

```
Text: "I love machine learning"
Character tokens: ['i', ' ', 'l', 'o', 'v', 'e', ...]
Word tokens: ['i', 'love', 'machine', 'learning']
Vocabulary size: 500
```

### NER Results

```
Text: "John works at Google in New York"
Tags: [B-PER, O, O, B-ORG, O, B-LOC, I-LOC]
```

### Sentiment Analysis

```
Text: "This product is amazing!"
Sentiment: Positive (confidence: 0.95)

Text: "I hate this service"
Sentiment: Negative (confidence: 0.87)
```

---

## 💡 Extension Ideas

1. **Sequence-to-Sequence Models**
   - Machine translation
   - Text summarization
   - Question answering

2. **Advanced Embeddings**
   - Word2Vec
   - GloVe
   - Contextual embeddings (BERT)

3. **Multi-task Learning**
   - Joint NER and sentiment
   - Shared embeddings
   - Transfer learning

---

## ✅ Success Criteria

- ✅ Implement multiple tokenization methods
- ✅ Create and use word embeddings
- ✅ Build NER system
- ✅ Classify sentiment accurately
- ✅ Create end-to-end pipeline

---

**Ready to build NLP applications? Let's process text!** 🚀
