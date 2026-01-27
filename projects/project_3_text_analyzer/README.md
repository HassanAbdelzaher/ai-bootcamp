# Project 3: Text Analyzer

> **Build text classification and generation systems using RNNs**

**Difficulty**: ⭐⭐ Intermediate  
**Time**: 3-4 hours  
**Prerequisites**: Steps 0-7 (Including RNN concepts, especially Step 7a: Text Generator)

---

## 📚 Table of Contents

1. [Project Overview](#project-overview)
2. [Problem 1: Text Classification](#problem-1-text-classification)
3. [Problem 2: Text Generation](#problem-2-text-generation)
4. [Key Concepts](#key-concepts)
5. [Step-by-Step Guide](#step-by-step-guide)
6. [Expected Results](#expected-results)
7. [Extension Ideas](#extension-ideas)
8. [Troubleshooting](#troubleshooting)

---

## 🎯 Project Overview

This project teaches you to work with **text data** using RNNs. You'll build two systems:

1. **Text Classifier** - Categorize text documents (e.g., news categories, sentiment)
2. **Text Generator** - Generate new text character by character

### Why Text Processing?

- **Real-world applications**: Email filtering, chatbots, content generation
- **Sequence understanding**: Learn how RNNs handle sequential data
- **Foundation for NLP**: Prepares you for advanced NLP tasks

---

## 📋 Problem 1: Text Classification

### Overview

Classify text documents into categories (e.g., news categories, sentiment classes). This demonstrates how RNNs can understand and categorize text.

### Learning Objectives

- Preprocess text data
- Extract features from text
- Build RNN-based classifier
- Evaluate classification performance
- Handle variable-length sequences

### Dataset Description

Create synthetic text data with different categories:

| Category | Example Features | Label |
|----------|-----------------|-------|
| **Sports** | "game", "team", "score", "win" | 0 |
| **Technology** | "computer", "software", "code" | 1 |
| **Science** | "research", "experiment", "data" | 2 |
| **News** | "report", "announce", "update" | 3 |

### Step-by-Step Implementation

#### Step 1: Create Text Dataset

```python
import numpy as np

# Define category keywords
categories = {
    0: ["game", "team", "player", "score", "win", "match"],
    1: ["computer", "software", "code", "program", "tech", "digital"],
    2: ["research", "experiment", "study", "data", "science", "lab"],
    3: ["report", "announce", "update", "news", "event", "story"]
}

def generate_text(category, length=50):
    """Generate synthetic text for a category"""
    keywords = categories[category]
    text = []
    
    # Add some keywords
    for _ in range(length // 10):
        text.append(np.random.choice(keywords))
    
    # Add random words
    all_words = ["the", "a", "an", "is", "are", "was", "were"]
    for _ in range(length - len(text)):
        text.append(np.random.choice(all_words))
    
    return " ".join(text)

# Generate dataset
texts = []
labels = []
for category in range(4):
    for _ in range(50):
        texts.append(generate_text(category))
        labels.append(category)
```

#### Step 2: Character-Level Encoding

```python
# Create character vocabulary
all_chars = set()
for text in texts:
    all_chars.update(text.lower())

char_to_idx = {char: idx for idx, char in enumerate(sorted(all_chars))}
idx_to_char = {idx: char for char, idx in char_to_idx.items()}
vocab_size = len(char_to_idx)

print(f"Vocabulary size: {vocab_size}")
print(f"Characters: {sorted(all_chars)[:20]}...")
```

#### Step 3: Convert Text to Sequences

```python
def text_to_sequence(text, max_length=100):
    """Convert text to sequence of character indices"""
    text = text.lower()
    sequence = [char_to_idx.get(char, 0) for char in text]
    
    # Pad or truncate to max_length
    if len(sequence) < max_length:
        sequence += [0] * (max_length - len(sequence))
    else:
        sequence = sequence[:max_length]
    
    return sequence

# Convert all texts
X = np.array([text_to_sequence(text) for text in texts])
y = np.array(labels)
```

#### Step 4: Build RNN Classifier

```python
import torch
import torch.nn as nn

class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, num_classes=4):
        super(TextClassifier, self).__init__()
        
        # Embedding layer: converts character indices to vectors
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # RNN layer: processes sequence
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        
        # Classifier: converts RNN output to class probabilities
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        # x shape: (batch, sequence_length)
        
        # Embed characters
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        
        # Process with RNN
        rnn_out, hidden = self.rnn(embedded)
        
        # Use last output
        last_output = rnn_out[:, -1, :]  # (batch, hidden_dim)
        
        # Classify
        output = self.fc(last_output)  # (batch, num_classes)
        return output
```

#### Step 5: Train Model

```python
model = TextClassifier(vocab_size, num_classes=4)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Convert to tensors
X_tensor = torch.LongTensor(X)
y_tensor = torch.LongTensor(y)

# Training loop
epochs = 100
for epoch in range(epochs):
    # Forward pass
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
```

---

## 📋 Problem 2: Text Generation

### Overview

Generate new text character by character using an RNN. The model learns patterns from training text and generates similar text.

### Learning Objectives

- Build character-level RNN
- Understand sequence generation
- Implement sampling strategies
- Control generation with temperature

### Dataset Description

Use a simple text corpus (poems, quotes, etc.):

```python
training_text = """
The quick brown fox jumps over the lazy dog.
Machine learning is fascinating and powerful.
Python is a great programming language.
Artificial intelligence will change the world.
"""
```

### Step-by-Step Implementation

#### Step 1: Prepare Text Data

```python
# Create character vocabulary
chars = sorted(set(training_text.lower()))
char_to_idx = {char: idx for idx, char in enumerate(chars)}
idx_to_char = {idx: char for char, idx in char_to_idx.items()}
vocab_size = len(chars)

print(f"Vocabulary: {chars}")
print(f"Vocabulary size: {vocab_size}")
```

#### Step 2: Create Sequences

```python
def create_sequences(text, seq_length=30):
    """Create input-target pairs for training"""
    X = []
    y = []
    
    for i in range(len(text) - seq_length):
        # Input: sequence of characters
        seq_in = text[i:i+seq_length]
        # Target: next character
        seq_out = text[i+seq_length]
        
        X.append([char_to_idx[char] for char in seq_in])
        y.append(char_to_idx[seq_out])
    
    return np.array(X), np.array(y)

X, y = create_sequences(training_text.lower(), seq_length=30)
print(f"Created {len(X)} sequences")
```

#### Step 3: Build Text Generator

```python
class TextGenerator(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128):
        super(TextGenerator, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        rnn_out, hidden = self.rnn(embedded, hidden)
        output = self.fc(rnn_out)
        return output, hidden
```

#### Step 4: Train Generator

```python
model = TextGenerator(vocab_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

X_tensor = torch.LongTensor(X)
y_tensor = torch.LongTensor(y)

epochs = 100
for epoch in range(epochs):
    outputs, _ = model(X_tensor)
    loss = criterion(outputs.reshape(-1, vocab_size), y_tensor)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
```

#### Step 5: Generate Text

```python
def generate_text(model, start_text, length=100, temperature=1.0):
    """Generate text starting from start_text"""
    model.eval()
    generated = start_text.lower()
    
    # Convert start text to sequence
    input_seq = torch.LongTensor([[char_to_idx[char] for char in generated[-30:]]])
    hidden = None
    
    for _ in range(length):
        # Forward pass
        output, hidden = model(input_seq, hidden)
        
        # Get probabilities
        logits = output[0, -1, :] / temperature
        probs = F.softmax(logits, dim=0)
        
        # Sample next character
        next_char_idx = torch.multinomial(probs, 1).item()
        next_char = idx_to_char[next_char_idx]
        
        # Append to generated text
        generated += next_char
        
        # Update input for next iteration
        input_seq = torch.LongTensor([[next_char_idx]])
    
    return generated

# Generate text
start = "the quick"
generated = generate_text(model, start, length=100, temperature=0.8)
print(f"Generated: {generated}")
```

**Temperature Explanation**:
- **Temperature < 1.0**: More conservative, predictable text
- **Temperature = 1.0**: Balanced randomness
- **Temperature > 1.0**: More creative, unpredictable text

---

## 🧠 Key Concepts

### 1. Character-Level vs Word-Level

**Character-Level** (This Project):
- Pros: Handles any text, smaller vocabulary
- Cons: Longer sequences, harder to learn

**Word-Level**:
- Pros: Faster training, better semantics
- Cons: Larger vocabulary, unknown words

### 2. RNN Architecture for Text

```
Input: Character sequence
  ↓
Embedding: Convert to vectors
  ↓
RNN: Process sequence step by step
  ↓
Output: Next character probabilities
```

### 3. Sequence Generation

**Process**:
1. Start with seed text
2. Predict next character
3. Append to text
4. Use new text to predict next character
5. Repeat

**Sampling Strategies**:
- **Greedy**: Always pick highest probability
- **Random**: Sample from probability distribution
- **Temperature**: Control randomness

---

## 📊 Expected Results

### Text Classification

```
Training Text Classifier...
Epoch 20/100, Loss: 1.2345
Epoch 40/100, Loss: 0.8765
...
Epoch 100/100, Loss: 0.1234

Evaluation:
Accuracy: 92.5%

Confusion Matrix:
        Sports  Tech  Science  News
Sports    48     1      0      1
Tech       2    47      1      0
Science    1     0     48      1
News       1     0      1     48
```

### Text Generation

```
Training Text Generator...
Epoch 20/100, Loss: 2.3456
Epoch 40/100, Loss: 1.8765
...
Epoch 100/100, Loss: 0.5432

Generating text...
Seed: "the quick"
Generated: "the quick brown fox jumps over the lazy dog. machine learning is fascinating and powerful. python is a great programming language..."
```

---

## 💡 Extension Ideas

### Beginner Extensions

1. **Try Different Sequence Lengths**
   - Compare seq_length = 20, 30, 50
   - Observe impact on generation quality

2. **Experiment with Temperature**
   - Try temperature = 0.5, 1.0, 1.5, 2.0
   - See how it affects creativity

3. **Add More Training Data**
   - Use longer text corpus
   - Compare generation quality

### Intermediate Extensions

4. **Use LSTM/GRU**
   - Replace RNN with LSTM
   - Compare performance
   - Better long-term memory

5. **Word-Level Generation**
   - Switch to word-level instead of character
   - Larger vocabulary but better semantics

6. **Multi-Class Classification**
   - Classify into more categories
   - Handle imbalanced classes

### Advanced Extensions

7. **Attention Mechanism**
   - Add attention to RNN
   - Better understanding of long sequences

8. **Transformer Architecture**
   - Implement simple transformer
   - Compare with RNN

9. **Real Dataset**
   - Use real text datasets
   - Handle preprocessing challenges

---

## 🔧 Troubleshooting

### Common Issues

**Issue 1: Generated text is repetitive**
- **Solution**: Increase temperature
- **Solution**: Train for more epochs
- **Solution**: Use larger model

**Issue 2: Model doesn't learn patterns**
- **Solution**: Check data preprocessing
- **Solution**: Verify character encoding
- **Solution**: Increase model capacity

**Issue 3: Out of memory**
- **Solution**: Reduce batch size
- **Solution**: Use shorter sequences
- **Solution**: Process in batches

---

## ✅ Success Criteria

- ✅ Text classifier achieves >85% accuracy
- ✅ Text generator produces coherent text
- ✅ Generated text shows learned patterns
- ✅ Code handles variable-length sequences
- ✅ Temperature control works correctly

---

## 🎓 Learning Outcomes

By completing this project, you will:

- ✅ Understand text preprocessing
- ✅ Build RNN-based text models
- ✅ Implement sequence generation
- ✅ Handle character-level encoding
- ✅ Control generation with temperature
- ✅ Evaluate text classification performance

---

## 📖 Additional Resources

- **Step 7 Documentation**: `docs/Step_7_RNNs.md`
- **Step 7a Documentation**: `docs/Step_7a_Text_Generator.md`
- **Step 7e Documentation**: `docs/Step_7e_NLP_Applications.md`

---

**Ready to work with text? Let's build intelligent text systems!** 🚀

**Next Steps**: After completing this project, move on to **Project 4: Image Classifier** to learn about CNNs.
