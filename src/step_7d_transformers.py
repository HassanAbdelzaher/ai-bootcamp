"""
Step 7d — Transformers (BERT, GPT)
Goal: Introduction to Transformer architecture and attention mechanism.
Tools: Python + PyTorch + NumPy
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore', message='.*Failed to initialize NumPy.*')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from plotting import plot_learning_curve

print("=== Step 7d: Transformers (BERT, GPT) ===")
print("Introduction to Transformer architecture and attention")
print()

# 7d.1 The Evolution: RNN → LSTM → Transformer
print("=== 7d.1 The Evolution ===")
print("RNN (1980s):")
print("  - Sequential processing")
print("  - Vanishing gradients")
print()
print("LSTM (1997):")
print("  - Solved vanishing gradients")
print("  - Better memory")
print("  - Still sequential")
print()
print("Transformer (2017):")
print("  - Parallel processing")
print("  - Attention mechanism")
print("  - No recurrence needed!")
print("  - Powers GPT, BERT, ChatGPT")
print()

# 7d.2 What is Attention?
print("=== 7d.2 What is Attention? ===")
print("Attention = Focus on relevant parts")
print()
print("Example: 'The cat sat on the mat'")
print("  When processing 'mat', attention focuses on:")
print("    - 'cat' (who sat)")
print("    - 'sat' (action)")
print("    - 'on' (position)")
print()
print("Self-Attention:")
print("  - Each word attends to all other words")
print("  - Learns relationships")
print("  - Parallel computation")
print()

# 7d.3 Simple Attention Implementation
print("=== 7d.3 Simple Attention Implementation ===")
def simple_attention(Q, K, V):
    """
    Simple attention mechanism
    Q = Query (what we're looking for)
    K = Key (what we're matching against)
    V = Value (what we retrieve)
    """
    # Compute attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1))
    scores = scores / np.sqrt(Q.size(-1))  # Scale
    attention_weights = F.softmax(scores, dim=-1)
    
    # Apply attention to values
    output = torch.matmul(attention_weights, V)
    return output, attention_weights

# Example
batch_size, seq_len, d_model = 2, 4, 8
Q = torch.randn(batch_size, seq_len, d_model)
K = torch.randn(batch_size, seq_len, d_model)
V = torch.randn(batch_size, seq_len, d_model)

output, weights = simple_attention(Q, K, V)
print(f"Input shape: {Q.shape}")
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {weights.shape}")
print("Attention weights show how much each position focuses on others")
print()

# 7d.4 Transformer Architecture Overview
print("=== 7d.4 Transformer Architecture Overview ===")
print("Transformer consists of:")
print("  1. Encoder (for understanding)")
print("  2. Decoder (for generation)")
print("  3. Multi-Head Attention")
print("  4. Position Encoding")
print("  5. Feed-Forward Networks")
print()

# 7d.5 Simple Transformer Block
print("=== 7d.5 Simple Transformer Block ===")
class SimpleTransformerBlock(nn.Module):
    """Simplified Transformer block"""
    def __init__(self, d_model=64, nhead=4):
        super(SimpleTransformerBlock, self).__init__()
        self.d_model = d_model
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        
        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # Self-attention with residual connection
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Feed-forward with residual connection
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        return x

# Test transformer block
block = SimpleTransformerBlock(d_model=64, nhead=4)
test_input = torch.randn(2, 10, 64)  # (batch, seq_len, d_model)
output = block(test_input)

print(f"Input shape: {test_input.shape}")
print(f"Output shape: {output.shape}")
print("Transformer block processes entire sequence in parallel!")
print()

# 7d.6 Position Encoding
print("=== 7d.6 Position Encoding ===")
print("Problem: Attention has no notion of position")
print("Solution: Add position encoding to embeddings")
print()

def positional_encoding(seq_len, d_model):
    """Generate positional encodings"""
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len).unsqueeze(1).float()
    
    div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                        -(np.log(10000.0) / d_model))
    
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    return pe

pos_enc = positional_encoding(10, 64)
print(f"Position encoding shape: {pos_enc.shape}")
print("Each position has a unique encoding")
print()

# 7d.7 BERT Overview
print("=== 7d.7 BERT Overview ===")
print("BERT = Bidirectional Encoder Representations from Transformers")
print()
print("Key features:")
print("  ✅ Bidirectional: Reads left-to-right AND right-to-left")
print("  ✅ Pre-trained on massive text corpus")
print("  ✅ Fine-tuned for specific tasks")
print()
print("Applications:")
print("  - Text classification")
print("  - Question answering")
print("  - Named entity recognition")
print("  - Sentiment analysis")
print()

# 7d.8 GPT Overview
print("=== 7d.8 GPT Overview ===")
print("GPT = Generative Pre-trained Transformer")
print()
print("Key features:")
print("  ✅ Autoregressive: Generates text one token at a time")
print("  ✅ Unidirectional: Reads left-to-right")
print("  ✅ Pre-trained then fine-tuned")
print()
print("Applications:")
print("  - Text generation")
print("  - Chatbots")
print("  - Code generation")
print("  - Creative writing")
print()

# 7d.9 BERT vs GPT
print("=== 7d.9 BERT vs GPT ===")
print("BERT:")
print("  ✅ Better for understanding tasks")
print("  ✅ Sees full context (bidirectional)")
print("  ✅ Great for classification")
print("  ❌ Not good for generation")
print()
print("GPT:")
print("  ✅ Better for generation tasks")
print("  ✅ Autoregressive generation")
print("  ✅ Great for creative tasks")
print("  ❌ Only sees left context")
print()

# 7d.10 Simple Task: Next Token Prediction
print("=== 7d.10 Simple Task: Next Token Prediction ===")
# Simple sequence prediction task
seq_length = 10
vocab_size = 20

# Create simple data
X = torch.randint(0, vocab_size, (100, seq_length))
y = torch.randint(0, vocab_size, (100,))

# Embedding + Transformer
class SimpleTransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=4):
        super(SimpleTransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = SimpleTransformerBlock(d_model, nhead)
        self.fc = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x[:, -1, :]  # Use last position
        return self.fc(x)

model = SimpleTransformerModel(vocab_size, d_model=64, nhead=4)
print("Model architecture:")
print(model)
print()

# 7d.11 Training
print("=== 7d.11 Training Simple Transformer ===")
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

losses = []
num_epochs = 50

print(f"Training for {num_epochs} epochs...")
for epoch in range(num_epochs):
    predictions = model(X)
    loss = loss_fn(predictions, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

print("Training complete!")
print()

# 7d.12 Learning Curve
print("=== 7d.12 Learning Curve ===")
plot_learning_curve(losses, title="Transformer Training Loss", ylabel="Loss (Cross-Entropy)")

# 7d.13 Why Transformers are Revolutionary
print("=== 7d.13 Why Transformers are Revolutionary ===")
print("✅ Parallel processing:")
print("   - Process entire sequence at once")
print("   - Much faster than RNNs")
print()
print("✅ Long-range dependencies:")
print("   - Attention connects any two positions")
print("   - No information loss over distance")
print()
print("✅ Scalability:")
print("   - Can handle very long sequences")
print("   - Powers large language models")
print()
print("✅ Transfer learning:")
print("   - Pre-train on large corpus")
print("   - Fine-tune for specific tasks")
print()

# 7d.14 Real-World Impact
print("=== 7d.14 Real-World Impact ===")
print("Transformers power:")
print("  🤖 ChatGPT (GPT-4)")
print("  🔍 Google Search (BERT)")
print("  🌐 GitHub Copilot (Code generation)")
print("  🎨 DALL-E (Image generation)")
print("  🗣️  Voice assistants")
print("  📝 Translation services")
print()

# 7d.15 Next Steps
print("=== 7d.15 Next Steps ===")
print("You've learned:")
print("  ✅ What attention mechanism is")
print("  ✅ How Transformers work")
print("  ✅ BERT vs GPT differences")
print("  ✅ Why Transformers are powerful")
print()
print("Try these next:")
print("  - Use Hugging Face transformers library")
print("  - Fine-tune BERT for your task")
print("  - Generate text with GPT")
print("  - Explore vision transformers (ViT)")
print("  - Learn about large language models (LLMs)")
print()

print("🎉 Transformer learning complete!")
print("You now understand the architecture behind ChatGPT and modern AI!")
