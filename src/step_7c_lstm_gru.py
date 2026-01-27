"""
Step 7c — LSTMs and GRUs (Advanced RNNs)
Goal: Learn about LSTM and GRU, improvements over basic RNNs.
Tools: Python + PyTorch + NumPy
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore', message='.*Failed to initialize NumPy.*')

import torch
import torch.nn as nn
import torch.optim as optim
from plotting import plot_learning_curve

print("=== Step 7c: LSTMs and GRUs ===")
print("Learning about advanced RNN architectures")
print()

# 7c.1 The Problem with Basic RNNs
print("=== 7c.1 The Problem with Basic RNNs ===")
print("Basic RNNs have limitations:")
print("  ❌ Vanishing gradient problem")
print("  ❌ Can't remember long sequences")
print("  ❌ Struggles with dependencies far apart")
print()
print("Example: 'The cat, which I saw yesterday, was...'")
print("  RNN might forget 'cat' by the time it reaches 'was'")
print()

# 7c.2 What is LSTM?
print("=== 7c.2 What is LSTM? ===")
print("LSTM = Long Short-Term Memory")
print()
print("Key innovation: Gated architecture")
print("  ✅ Forget gate: What to forget")
print("  ✅ Input gate: What to remember")
print("  ✅ Output gate: What to output")
print("  ✅ Cell state: Long-term memory")
print()
print("Benefits:")
print("  ✅ Can remember information for long periods")
print("  ✅ Solves vanishing gradient problem")
print("  ✅ Better at long sequences")
print()

# 7c.3 What is GRU?
print("=== 7c.3 What is GRU? ===")
print("GRU = Gated Recurrent Unit")
print()
print("Simplified version of LSTM:")
print("  ✅ Reset gate: How much past to forget")
print("  ✅ Update gate: How much new info to add")
print()
print("Benefits:")
print("  ✅ Simpler than LSTM (fewer parameters)")
print("  ✅ Often performs similarly to LSTM")
print("  ✅ Faster to train")
print()

# 7c.4 Create Long Sequence Data
print("=== 7c.4 Create Long Sequence Data ===")
def create_long_sequence_data(seq_length=50, num_samples=100):
    """Create sequences with long dependencies"""
    # Lists to store input sequences and target values
    X = []  # Input sequences
    y = []  # Target values (depends on first number, not last!)
    
    for _ in range(num_samples):
        # Create pattern: first number determines target (long dependency!)
        # This tests if model can remember information from far back
        # np.random.randint(1, 5) returns random integer in [1, 5)
        start = np.random.randint(1, 5)
        sequence = []
        
        # Add random numbers in middle
        # This makes it harder - lots of noise between first and last
        for i in range(seq_length - 1):
            if i == 0:
                # First number: the important one we need to remember
                sequence.append(start)
            else:
                # Middle numbers: random noise (distraction)
                # np.random.randint(1, 10) returns random integer in [1, 10)
                sequence.append(np.random.randint(1, 10))
        
        # Last number (target) depends on FIRST number, not last!
        # This is the challenge: model must remember first number across long sequence
        # start % 2 == 0: Check if start is even
        if start % 2 == 0:
            target = 0  # Even first number → target is 0
        else:
            target = 1  # Odd first number → target is 1
        
        # Store sequence and target
        X.append(sequence)
        y.append(target)
    
    # Convert to NumPy arrays
    # X: float32 for input features
    # y: int64 for classification labels
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

X, y = create_long_sequence_data(seq_length=30, num_samples=200)
print(f"Created {len(X)} sequences of length {X.shape[1]}")
print(f"Task: Remember first number to predict last (long dependency!)")
print()

# Convert to tensors
X_tensor = torch.FloatTensor(X).unsqueeze(-1)  # (batch, seq_length, features)
y_tensor = torch.LongTensor(y)  # (batch,)

print(f"Tensor shapes: X={X_tensor.shape}, y={y_tensor.shape}")
print()

# 7c.5 Compare RNN, LSTM, and GRU
print("=== 7c.5 Compare RNN, LSTM, and GRU ===")

# Build three models
class SimpleRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_classes=2):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        return self.fc(out)

class SimpleLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_classes=2):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

class SimpleGRU(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_classes=2):
        super(SimpleGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return self.fc(out)

rnn_model = SimpleRNN()
lstm_model = SimpleLSTM()
gru_model = SimpleGRU()

print("Model architectures:")
print(f"  RNN parameters: {sum(p.numel() for p in rnn_model.parameters())}")
print(f"  LSTM parameters: {sum(p.numel() for p in lstm_model.parameters())}")
print(f"  GRU parameters: {sum(p.numel() for p in gru_model.parameters())}")
print()

# 7c.6 Train LSTM
print("=== 7c.6 Training LSTM ===")
model = lstm_model
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

losses = []
num_epochs = 100

print(f"Training LSTM for {num_epochs} epochs...")
for epoch in range(num_epochs):
    predictions = model(X_tensor)
    loss = loss_fn(predictions, y_tensor)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    if (epoch + 1) % 25 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

print("Training complete!")
print()

# 7c.7 Learning Curve
print("=== 7c.7 Learning Curve ===")
plot_learning_curve(losses, title="LSTM Training Loss", ylabel="Loss (Cross-Entropy)")

# 7c.8 Evaluate Performance
print("=== 7c.8 Evaluate Performance ===")
model.eval()
with torch.no_grad():
    predictions = model(X_tensor[:20])
    predicted_classes = torch.argmax(predictions, dim=1)
    
    correct = (predicted_classes == y_tensor[:20]).sum().item()
    accuracy = correct / 20 * 100
    
    print(f"Accuracy on 20 samples: {accuracy:.1f}%")
    print(f"Correct: {correct}/20")
print()

# 7c.9 When to Use Each
print("=== 7c.9 When to Use Each Architecture ===")
print("Basic RNN:")
print("  ✅ Simple tasks")
print("  ✅ Short sequences")
print("  ✅ Fast training")
print("  ❌ Long sequences")
print()
print("LSTM:")
print("  ✅ Long sequences")
print("  ✅ Complex dependencies")
print("  ✅ When memory is critical")
print("  ❌ More parameters")
print()
print("GRU:")
print("  ✅ Good balance of performance and speed")
print("  ✅ Often as good as LSTM")
print("  ✅ Fewer parameters than LSTM")
print("  ✅ Faster training")
print()

# 7c.10 LSTM Internal Structure
print("=== 7c.10 LSTM Internal Structure ===")
print("LSTM has 4 main components:")
print("  1. Forget Gate: f_t = σ(W_f · [h_{t-1}, x_t] + b_f)")
print("  2. Input Gate: i_t = σ(W_i · [h_{t-1}, x_t] + b_i)")
print("  3. Cell State: C_t = f_t * C_{t-1} + i_t * C̃_t")
print("  4. Output Gate: o_t = σ(W_o · [h_{t-1}, x_t] + b_o)")
print()
print("This allows LSTM to:")
print("  - Remember information selectively")
print("  - Forget irrelevant information")
print("  - Update with new information")
print()

# 7c.11 Real-World Applications
print("=== 7c.11 Real-World Applications ===")
print("LSTM/GRU are used in:")
print("  📝 Machine translation (Google Translate)")
print("  💬 Chatbots and conversational AI")
print("  📊 Time series forecasting")
print("  🎵 Music generation")
print("  📖 Text summarization")
print("  🎬 Video analysis")
print()

# 7c.12 Next Steps
print("=== 7c.12 Next Steps ===")
print("You've learned:")
print("  ✅ Why basic RNNs have limitations")
print("  ✅ How LSTM solves the vanishing gradient problem")
print("  ✅ How GRU is a simpler alternative")
print("  ✅ When to use each architecture")
print()
print("Try these next:")
print("  - Compare RNN vs LSTM vs GRU on same task")
print("  - Use LSTM for text generation")
print("  - Explore bidirectional LSTM")
print("  - Learn about Transformers (next step!)")
print()

print("🎉 LSTM and GRU learning complete!")
