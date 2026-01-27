"""
Project 7d: Transformers and Attention
Implement Transformer architecture with attention mechanism
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import sys
import os

# Add parent directory to path to import plotting utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))
from plotting import plot_learning_curve

print("=" * 70)
print("Project 7d: Transformers and Attention")
print("=" * 70)
print()

# ============================================================================
# Step 1: Self-Attention Implementation
# ============================================================================
print("=" * 70)
print("Step 1: Implementing Self-Attention")
print("=" * 70)
print()

class SelfAttention(nn.Module):
    def __init__(self, d_model, d_k=None, d_v=None):
        # super(): Call parent class constructor
        super(SelfAttention, self).__init__()
        
        # d_model: Dimension of input embeddings
        # d_k: Dimension of Query and Key vectors (defaults to d_model)
        # d_v: Dimension of Value vectors (defaults to d_model)
        if d_k is None:
            d_k = d_model
        if d_v is None:
            d_v = d_model
        
        # Store dimension for scaling
        self.d_k = d_k
        
        # Linear layers to compute Query, Key, Value
        # W_q: Projects input to Query space
        # W_k: Projects input to Key space
        # W_v: Projects input to Value space
        # All transform from d_model to their respective dimensions
        self.W_q = nn.Linear(d_model, d_k)
        self.W_k = nn.Linear(d_model, d_k)
        self.W_v = nn.Linear(d_model, d_v)
    
    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        # batch: Number of samples in batch
        # seq_len: Length of sequence (e.g., number of words)
        # d_model: Dimension of each element in sequence
        batch_size, seq_len, d_model = x.size()
        
        # ===== STEP 1: COMPUTE Q, K, V =====
        # Query (Q): What we're looking for
        # Key (K): What we're matching against
        # Value (V): What we retrieve
        # All computed from same input x (self-attention)
        Q = self.W_q(x)  # (batch, seq_len, d_k) - Query vectors
        K = self.W_k(x)  # (batch, seq_len, d_k) - Key vectors
        V = self.W_v(x)  # (batch, seq_len, d_v) - Value vectors
        
        # ===== STEP 2: COMPUTE ATTENTION SCORES =====
        # Calculate similarity between queries and keys
        # torch.matmul(Q, K.transpose(-2, -1)): Matrix multiplication
        # Q: (batch, seq_len, d_k)
        # K.transpose(-2, -1): Transpose last two dims → (batch, d_k, seq_len)
        # Result: (batch, seq_len, seq_len) - similarity scores
        # Each element [i, j] = similarity between query i and key j
        scores = torch.matmul(Q, K.transpose(-2, -1))
        
        # Scale scores to prevent extreme values
        # Divide by sqrt(d_k) - standard scaling in attention
        # Prevents softmax from becoming too peaked (all probability on one element)
        # math.sqrt(self.d_k): Square root of key dimension
        scores = scores / math.sqrt(self.d_k)
        
        # Convert scores to probabilities using softmax
        # F.softmax(scores, dim=-1): Apply softmax along last dimension
        # Result: (batch, seq_len, seq_len) - attention weights
        # Each row sums to 1.0 (probabilities)
        # Higher score = more attention to that position
        attention_weights = F.softmax(scores, dim=-1)
        
        # ===== STEP 3: APPLY ATTENTION TO VALUES =====
        # Weighted sum of values based on attention weights
        # torch.matmul(attention_weights, V): Matrix multiplication
        # attention_weights: (batch, seq_len, seq_len) - how much to attend
        # V: (batch, seq_len, d_v) - values to retrieve
        # Result: (batch, seq_len, d_v) - attended output
        # Each output position is weighted combination of all value positions
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights

# Test self-attention
print("Testing Self-Attention...")
batch_size, seq_len, d_model = 2, 4, 8
x = torch.randn(batch_size, seq_len, d_model)
attention = SelfAttention(d_model)
output, weights = attention(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {weights.shape}")
print()

# ============================================================================
# Step 2: Multi-Head Attention
# ============================================================================
print("=" * 70)
print("Step 2: Implementing Multi-Head Attention")
print("=" * 70)
print()

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

print("Testing Multi-Head Attention...")
multi_head = MultiHeadAttention(d_model=64, num_heads=4)
output, weights = multi_head(x)
print(f"Multi-head output shape: {output.shape}")
print()

# ============================================================================
# Step 3: Positional Encoding
# ============================================================================
print("=" * 70)
print("Step 3: Implementing Positional Encoding")
print("=" * 70)
print()

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        # super(): Call parent class constructor
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        # pe: (max_len, d_model) - encoding for each position
        # max_len: Maximum sequence length we can handle
        pe = torch.zeros(max_len, d_model)
        
        # position: Position indices [0, 1, 2, ..., max_len-1]
        # torch.arange(0, max_len): Creates [0, 1, 2, ..., max_len-1]
        # .unsqueeze(1): Add dimension → (max_len, 1) for broadcasting
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # div_term: Division term for frequency calculation
        # torch.arange(0, d_model, 2): [0, 2, 4, ..., d_model-2] (even indices)
        # * (-math.log(10000.0) / d_model): Frequency scaling factor
        # torch.exp(...): Exponential to get frequency values
        # This creates different frequencies for different dimensions
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        # Apply sine to even dimensions
        # pe[:, 0::2]: Select even columns (0, 2, 4, ...)
        # torch.sin(position * div_term): Sinusoidal encoding
        # Each dimension gets different frequency based on position
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cosine to odd dimensions
        # pe[:, 1::2]: Select odd columns (1, 3, 5, ...)
        # torch.cos(position * div_term): Cosine encoding
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension: (1, max_len, d_model)
        # This allows broadcasting when adding to input
        pe = pe.unsqueeze(0)
        
        # register_buffer: Register as buffer (not a parameter)
        # Buffers are saved with model but not updated during training
        # 'pe': Name of buffer
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        # Add positional encoding to input embeddings
        # self.pe[:, :x.size(1), :]: Select first seq_len positions
        # x.size(1): Sequence length of current input
        # Broadcasting: (batch, seq_len, d_model) + (1, seq_len, d_model)
        # Result: Each position gets its unique positional encoding added
        x = x + self.pe[:, :x.size(1), :]
        return x

print("Testing Positional Encoding...")
pos_encoding = PositionalEncoding(d_model=64)
encoded = pos_encoding(x)
print(f"Encoded shape: {encoded.shape}")
print()

# ============================================================================
# Step 4: Transformer Block
# ============================================================================
print("=" * 70)
print("Step 4: Building Transformer Block")
print("=" * 70)
print()

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

print("Testing Transformer Block...")
transformer_block = TransformerBlock(d_model=64, num_heads=4, d_ff=256)
output = transformer_block(x)
print(f"Transformer block output shape: {output.shape}")
print()

# ============================================================================
# Step 5: Complete Transformer Model
# ============================================================================
print("=" * 70)
print("Step 5: Building Complete Transformer Model")
print("=" * 70)
print()

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
        
        # Classification (mean pooling)
        x = x.mean(dim=1)  # Mean pooling
        output = self.classifier(x)
        
        return output

# ============================================================================
# Step 6: Training on Classification Task
# ============================================================================
print("=" * 70)
print("Step 6: Training Transformer on Classification")
print("=" * 70)
print()

# Create simple classification dataset
X, y = make_classification(
    n_samples=500, n_features=20, n_classes=2, random_state=42
)

# Convert to sequences (simulate text)
# Each feature becomes a "word" in our vocabulary
vocab_size = 100
X_sequences = (X * 10).astype(int) % vocab_size
X_sequences = np.clip(X_sequences, 0, vocab_size - 1)

# Pad sequences to same length
max_len = 20
X_padded = np.zeros((len(X_sequences), max_len), dtype=int)
for i, seq in enumerate(X_sequences):
    length = min(len(seq), max_len)
    X_padded[i, :length] = seq[:length]

X_train, X_test, y_train, y_test = train_test_split(
    X_padded, y, test_size=0.2, random_state=42
)

X_train_tensor = torch.LongTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)
X_test_tensor = torch.LongTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)

# Train model
model = TransformerClassifier(
    vocab_size=vocab_size, 
    d_model=64, 
    num_heads=4, 
    num_layers=2,
    num_classes=2
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Training Transformer...")
epochs = 50
losses = []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1}/{epochs}: Loss = {loss.item():.4f}")

print()

# Evaluation
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predictions = torch.max(outputs, 1)
    accuracy = (predictions == y_test_tensor).float().mean().item()

print(f"Test Accuracy: {accuracy:.2%}")
print()

# Visualize training
plot_learning_curve(losses, title="Transformer Training Loss")

print("=" * 70)
print("Project 7d Complete!")
print("=" * 70)
print()
print("Summary:")
print("  ✅ Implemented self-attention mechanism")
print("  ✅ Built multi-head attention")
print("  ✅ Added positional encoding")
print("  ✅ Created Transformer blocks")
print("  ✅ Trained Transformer classifier")
print()
