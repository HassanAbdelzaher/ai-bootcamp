"""
Step 7 — RNNs (Recurrent Neural Networks for Sequences)
Goal: Learn how to process sequences (text, time series) using Recurrent Neural Networks.
Tools: Python + PyTorch + NumPy
"""

# Import numpy first to ensure proper initialization before PyTorch
import numpy as np
import warnings

# Suppress NumPy initialization warnings (common with PyTorch)
warnings.filterwarnings('ignore', message='.*Failed to initialize NumPy.*')

import torch
import torch.nn as nn
import torch.optim as optim
from plotting import plot_learning_curve

# Check if PyTorch is available
try:
    print("PyTorch version:", torch.__version__)
except ImportError:
    print("ERROR: PyTorch is not installed!")
    print("Install with: pip install torch torchvision torchaudio")
    exit(1)
print()

# 7.1 Big Idea: Why RNNs?
print("=== 7.1 Big Idea: Why RNNs? ===")
print("Regular neural networks:")
print("  - Process one input → one output")
print("  - No memory of previous inputs")
print("")
print("RNNs (Recurrent Neural Networks):")
print("  - Process sequences (text, time series)")
print("  - Remember previous information")
print("  - Perfect for: language, music, stock prices, etc.")
print()

# 7.2 Simple Sequence Example: Number Prediction
print("=== 7.2 Simple Sequence Example ===")
print("Task: Predict the next number in a sequence")
print("Example: [1, 2, 3, 4] → next should be 5")
print()

# Create simple sequence data
def create_sequence_data(seq_length=10, num_samples=100):
    """Create simple sequences: each number is previous + 1"""
    X = []
    y = []
    
    for _ in range(num_samples):
        start = np.random.randint(1, 10)
        sequence = [start + i for i in range(seq_length)]
        X.append(sequence[:-1])  # Input: all but last
        y.append(sequence[-1])    # Target: last number
    
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

X, y = create_sequence_data(seq_length=5, num_samples=50)
print("Sample sequences:")
for i in range(3):
    print(f"  Input: {X[i]} → Target: {y[i]}")
print()

# 7.3 Preparing Data for RNN
print("=== 7.3 Preparing Data for RNN ===")
print("RNNs need data in shape: (batch, sequence_length, features)")
print("Our data shape:", X.shape, "→ (samples, sequence_length)")
print()

# Convert to PyTorch tensors
X_tensor = torch.FloatTensor(X).unsqueeze(-1)  # Add feature dimension: (50, 4, 1)
y_tensor = torch.FloatTensor(y).unsqueeze(-1)  # (50, 1)

print("Tensor shapes:")
print(f"  X: {X_tensor.shape} (batch, sequence_length, features)")
print(f"  y: {y_tensor.shape} (batch, output)")
print()

# 7.4 Building an RNN
print("=== 7.4 Building an RNN ===")
class SimpleRNN(nn.Module):
    """Simple RNN for sequence prediction"""
    def __init__(self, input_size=1, hidden_size=32, output_size=1):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        
        # RNN layer: processes sequence step by step
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        
        # Output layer: converts hidden state to prediction
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x shape: (batch, sequence_length, features)
        # RNN returns: output, hidden
        rnn_out, hidden = self.rnn(x)
        
        # Use the last output from the sequence
        last_output = rnn_out[:, -1, :]  # (batch, hidden_size)
        
        # Predict next value
        prediction = self.fc(last_output)
        return prediction

model = SimpleRNN(input_size=1, hidden_size=32, output_size=1)
print("Model architecture:")
print(model)
print()

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")
print()

# 7.5 Training the RNN
print("=== 7.5 Training the RNN ===")
loss_fn = nn.MSELoss()  # Mean Squared Error for regression
optimizer = optim.Adam(model.parameters(), lr=0.001)

losses = []
num_epochs = 500

print(f"Training for {num_epochs} epochs...")
for epoch in range(num_epochs):
    # Forward pass
    predictions = model(X_tensor)
    loss = loss_fn(predictions, y_tensor)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    # Print progress
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

print("Training complete!")
print()

# 7.6 Learning Curve
print("=== 7.6 Learning Curve ===")
plot_learning_curve(losses, title="RNN Training Loss", ylabel="Loss (MSE)")

# 7.7 Testing the RNN
print("=== 7.7 Testing the RNN ===")
model.eval()  # Set to evaluation mode
with torch.no_grad():
    test_predictions = model(X_tensor[:5])  # Test on first 5 samples
    
    print("Sample predictions:")
    for i in range(5):
        actual = y_tensor[i].item()
        predicted = test_predictions[i].item()
        error = abs(actual - predicted)
        print(f"  Sequence {X[i]} → Actual: {actual:.1f}, Predicted: {predicted:.2f}, Error: {error:.2f}")
print()

# 7.8 Text Sequence Example (Character-level)
print("=== 7.8 Text Sequence Example ===")
print("RNNs can also process text character by character")
print()

# Simple character-level example
text = "hello"
print(f"Text: '{text}'")
print("Character sequence:", list(text))

# Convert characters to numbers
char_to_idx = {char: idx for idx, char in enumerate(set(text))}
idx_to_char = {idx: char for char, idx in char_to_idx.items()}

print("Character to index mapping:", char_to_idx)
print()

# Create sequences: predict next character
def create_text_sequences(text, seq_length=3):
    """Create sequences from text"""
    X_text = []
    y_text = []
    
    for i in range(len(text) - seq_length):
        seq = [char_to_idx[char] for char in text[i:i+seq_length]]
        next_char = char_to_idx[text[i+seq_length]]
        X_text.append(seq)
        y_text.append(next_char)
    
    return np.array(X_text), np.array(y_text)

X_text, y_text = create_text_sequences(text, seq_length=3)
print("Text sequences:")
for i in range(len(X_text)):
    input_chars = ''.join([idx_to_char[idx] for idx in X_text[i]])
    target_char = idx_to_char[y_text[i]]
    print(f"  '{input_chars}' → '{target_char}'")
print()

# 7.9 Why RNNs Matter
print("=== 7.9 Why RNNs Matter ===")
print("✅ Process sequences of any length")
print("✅ Remember context from previous steps")
print("✅ Used in:")
print("   - Language translation")
print("   - Text generation")
print("   - Speech recognition")
print("   - Time series prediction")
print("   - Music generation")
print()

# 7.10 Limitations and Improvements
print("=== 7.10 Limitations and Improvements ===")
print("RNN limitations:")
print("  ❌ Can struggle with long sequences")
print("  ❌ Vanishing gradient problem")
print("")
print("Better alternatives:")
print("  ✅ LSTM (Long Short-Term Memory)")
print("  ✅ GRU (Gated Recurrent Unit)")
print("  ✅ Transformer (attention mechanism)")
print()

# 7.11 Next Steps
print("=== 7.11 Next Steps ===")
print("You've learned:")
print("  ✅ What RNNs are and why they're useful")
print("  ✅ How to build and train an RNN in PyTorch")
print("  ✅ How to process sequences")
print("")
print("Try these next:")
print("  - Build a text generator")
print("  - Predict stock prices")
print("  - Learn about LSTMs and GRUs")
print("  - Explore Transformers (BERT, GPT)")
print()

print("🎉 Congratulations on completing Step 7: RNNs!")
print("You now understand how AI processes sequences and text!")
