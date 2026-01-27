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

## 7a.6 Visualize Character Embeddings

### What are Embeddings?

**Embeddings** convert characters (or words) into dense vectors. They are learned during training and capture relationships between characters.

**Key properties:**
- Similar characters have similar embeddings
- Learned automatically during training
- Useful for understanding what the model learned

### Extracting Embeddings

```python
from plotting import plot_word_embeddings

# Extract embeddings from the trained model
model.eval()
embedding_weights = model.embedding.weight.data.cpu().numpy()

# Select a subset of characters to visualize
sample_chars = ['a', 'e', 'i', 'o', 'u', 't', 'n', 's', 'r', 'h', 'd', 'l', 'c', 'm', 'f', 'p']
sample_indices = [char_to_idx[char] for char in sample_chars if char in char_to_idx]
sample_embeddings = embedding_weights[sample_indices]
sample_words = [char for char in sample_chars if char in char_to_idx]

print(f"Visualizing embeddings for {len(sample_words)} characters")
print(f"Characters: {', '.join(sample_words)}")

# Visualize embeddings
plot_word_embeddings(sample_embeddings, words=sample_words, 
                    title="Character Embeddings Visualization (2D Projection)")
```

**Code Explanation:**
- `model.embedding.weight.data`: Get embedding weights from trained model
- `.cpu().numpy()`: Convert to NumPy array (for visualization)
- `sample_chars`: Select interesting characters (vowels, common consonants)
- `sample_indices`: Get indices for selected characters
- `plot_word_embeddings()`: Visualize in 2D using PCA

### Interpreting Embeddings

**What you'll see:**
- **Vowels (a, e, i, o, u)**: Might cluster together
- **Common consonants (t, n, s)**: Might be close to each other
- **Model learns relationships**: Similar characters have similar positions

**Observations:**
- Characters that appear together in text tend to have similar embeddings
- The model learns semantic relationships between characters
- Embeddings capture patterns in the training text

### Understanding the Visualization

The visualization uses **PCA (Principal Component Analysis)** to reduce high-dimensional embeddings to 2D for plotting.

**What PCA does:**
- Finds the most important directions in the embedding space
- Projects embeddings onto these directions
- Preserves as much information as possible in 2D

**Explained variance**: Shows how much of the original information is preserved (higher is better).

---

## 7a.7 Improvements

- Use word-level instead of character-level
- Train on larger datasets
- Use LSTM or GRU
- Try Transformer models (GPT)
