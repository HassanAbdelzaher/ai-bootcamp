"""
Step 7a — Text Generator (Character-level RNN)
Goal: Build a character-level text generator using RNNs.
Tools: Python + PyTorch + NumPy
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore', message='.*Failed to initialize NumPy.*')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from plotting import plot_learning_curve, plot_word_embeddings

print("=== Step 7a: Text Generator ===")
print("Building a character-level RNN to generate text")
print()

# 7a.1 Prepare Text Data
print("=== 7a.1 Prepare Text Data ===")
# Simple training text
text = """The quick brown fox jumps over the lazy dog.
Python is a great programming language for AI.
Neural networks can learn patterns in text.
Recurrent networks remember previous information.
Deep learning is transforming technology."""

print("Training text (first 100 chars):")
print(text[:100] + "...")
print(f"Total characters: {len(text)}")
print()

# Create character mappings
chars = sorted(list(set(text)))
char_to_idx = {char: idx for idx, char in enumerate(chars)}
idx_to_char = {idx: char for char, idx in char_to_idx.items()}

vocab_size = len(chars)
print(f"Vocabulary size: {vocab_size} unique characters")
print(f"Characters: {''.join(chars[:20])}...")
print()

# 7a.2 Create Sequences
print("=== 7a.2 Create Sequences ===")
def create_text_sequences(text, seq_length=20):
    """Create input-output pairs for character prediction"""
    X = []
    y = []
    
    for i in range(len(text) - seq_length):
        seq = text[i:i+seq_length]
        next_char = text[i+seq_length]
        
        X.append([char_to_idx[char] for char in seq])
        y.append(char_to_idx[next_char])
    
    return np.array(X), np.array(y)

seq_length = 20
X, y = create_text_sequences(text, seq_length)

print(f"Created {len(X)} sequences of length {seq_length}")
print(f"Example sequence: '{text[:seq_length]}' → '{text[seq_length]}'")
print()

# Convert to tensors
X_tensor = torch.LongTensor(X)
y_tensor = torch.LongTensor(y)

print(f"Tensor shapes: X={X_tensor.shape}, y={y_tensor.shape}")
print()

# 7a.3 Build Character-level RNN
print("=== 7a.3 Build Character-level RNN ===")
class CharRNN(nn.Module):
    """Character-level RNN for text generation"""
    def __init__(self, vocab_size, hidden_size=128, num_layers=2):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Embedding layer: converts character indices to vectors
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # RNN layer
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers, batch_first=True)
        
        # Output layer: predicts next character
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hidden=None):
        # x shape: (batch, seq_length)
        # Embedding: (batch, seq_length, hidden_size)
        x = self.embedding(x)
        
        # RNN: (batch, seq_length, hidden_size)
        out, hidden = self.rnn(x, hidden)
        
        # Use last output
        out = out[:, -1, :]  # (batch, hidden_size)
        
        # Predict next character
        out = self.fc(out)  # (batch, vocab_size)
        return out, hidden
    
    def init_hidden(self, batch_size):
        """Initialize hidden state"""
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)

model = CharRNN(vocab_size, hidden_size=128, num_layers=2)
print("Model architecture:")
print(model)
print()

# 7a.4 Training
print("=== 7a.4 Training the Text Generator ===")
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

losses = []
num_epochs = 200

print(f"Training for {num_epochs} epochs...")
for epoch in range(num_epochs):
    hidden = model.init_hidden(X_tensor.size(0))
    
    # Forward pass
    output, hidden = model(X_tensor, hidden)
    loss = loss_fn(output, y_tensor)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    # Clip gradients to prevent explosion
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
    optimizer.step()
    
    losses.append(loss.item())
    
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

print("Training complete!")
print()

# 7a.5 Learning Curve
print("=== 7a.5 Learning Curve ===")
plot_learning_curve(losses, title="Text Generator Training Loss", ylabel="Loss (Cross-Entropy)")

# 7a.6 Visualize Word Embeddings
print("=== 7a.6 Visualize Character Embeddings ===")
print("Embeddings convert characters into dense vectors")
print("  - Similar characters have similar embeddings")
print("  - Learned during training")
print("  - Useful for understanding what the model learned")
print()

# Extract embeddings from the trained model
# model.eval() sets model to evaluation mode (disables dropout, batch norm updates)
# This ensures consistent behavior when extracting embeddings
model.eval()

# Get embedding weights from the model
# model.embedding is the embedding layer (nn.Embedding)
# .weight accesses the weight tensor (vocab_size × embedding_dim)
# .data gets the underlying tensor data
# .cpu() moves tensor from GPU to CPU (if on GPU)
# .numpy() converts PyTorch tensor to NumPy array (for visualization)
embedding_weights = model.embedding.weight.data.cpu().numpy()
# Shape: (vocab_size, hidden_size) - each row is an embedding vector for one character

# Select a subset of characters to visualize
# Choose interesting characters: vowels and common consonants
# This makes visualization more meaningful than all characters
sample_chars = ['a', 'e', 'i', 'o', 'u', 't', 'n', 's', 'r', 'h', 'd', 'l', 'c', 'm', 'f', 'p']

# Get indices for selected characters
# List comprehension: for each char in sample_chars, get its index if it exists in char_to_idx
# char_to_idx is the dictionary mapping characters to indices
# Only include characters that exist in the vocabulary
sample_indices = [char_to_idx[char] for char in sample_chars if char in char_to_idx]

# Extract embeddings for selected characters
# sample_indices are row indices into embedding_weights
# This selects only the embedding vectors for our sample characters
sample_embeddings = embedding_weights[sample_indices]
# Shape: (len(sample_indices), hidden_size) - embeddings for selected characters

# Get list of characters that actually exist in vocabulary
# Filter sample_chars to only include those in char_to_idx
sample_words = [char for char in sample_chars if char in char_to_idx]

# Print information about what we're visualizing
# {len(sample_words)} gets number of characters being visualized
print(f"Visualizing embeddings for {len(sample_words)} characters")
# ', '.join(sample_words) creates comma-separated string of characters
print(f"Characters: {', '.join(sample_words)}")
print()

# Visualize embeddings
# plot_word_embeddings creates a 2D scatter plot of embeddings
# Uses PCA (Principal Component Analysis) to reduce high-dimensional embeddings to 2D
# sample_embeddings: The embedding vectors to visualize
# words: Character labels to display on plot
# title: Plot title
plot_word_embeddings(sample_embeddings, words=sample_words, 
                    title="Character Embeddings Visualization (2D Projection)")

print("Observations:")
print("  - Vowels (a, e, i, o, u) might cluster together")
print("  - Common consonants (t, n, s) might be close")
print("  - Model learns relationships between characters")
print()

# 7a.7 Generate Text
print("=== 7a.7 Generate Text ===")
def generate_text(model, start_text, length=100, temperature=0.8):
    """Generate text given a starting sequence"""
    model.eval()
    chars = list(start_text)
    
    # Convert start text to indices
    hidden = model.init_hidden(1)
    
    for char in start_text[:-1]:
        x = torch.LongTensor([[char_to_idx[char]]])
        _, hidden = model(x, hidden)
    
    # Generate characters
    for _ in range(length):
        x = torch.LongTensor([[char_to_idx[chars[-1]]]])
        output, hidden = model(x, hidden)
        
        # Apply temperature for diversity
        output = output / temperature
        probs = F.softmax(output, dim=1)
        
        # Sample from distribution
        next_idx = torch.multinomial(probs, 1).item()
        chars.append(idx_to_char[next_idx])
    
    return ''.join(chars)

# Test generation
print("Generating text...")
start_seed = "The quick brown"
generated = generate_text(model, start_seed, length=150, temperature=0.8)

print(f"\nStarting with: '{start_seed}'")
print(f"\nGenerated text:")
print(generated)
print()

# 7a.8 Tips for Better Text Generation
print("=== 7a.7 Tips for Better Text Generation ===")
print("✅ Use more training data (books, articles)")
print("✅ Train for more epochs")
print("✅ Use LSTM or GRU instead of RNN")
print("✅ Adjust temperature (lower = more conservative)")
print("✅ Use word-level instead of character-level")
print("✅ Try Transformer models (GPT, BERT)")
print()

print("🎉 Text generator complete!")
print("Try modifying the training text and see what it generates!")
