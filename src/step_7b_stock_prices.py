"""
Step 7b — Stock Price Prediction (Time Series RNN)
Goal: Predict stock prices using RNNs for time series data.
Tools: Python + PyTorch + NumPy + Matplotlib
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore', message='.*Failed to initialize NumPy.*')

import torch
import torch.nn as nn
import torch.optim as optim
from plotting import plot_learning_curve

print("=== Step 7b: Stock Price Prediction ===")
print("Using RNNs to predict future stock prices from historical data")
print()

# 7b.1 Generate Synthetic Stock Price Data
print("=== 7b.1 Generate Synthetic Stock Price Data ===")
def generate_stock_prices(days=200, start_price=100, volatility=0.02):
    """Generate synthetic stock price data using random walk model"""
    # Start with initial price
    prices = [start_price]
    
    # Generate prices for remaining days
    for _ in range(days - 1):
        # Random walk with slight upward trend
        # np.random.normal(mean, std) generates random value from normal distribution
        # mean=0.001: Slight upward trend (0.1% average daily increase)
        # std=volatility: Daily price volatility (2% standard deviation)
        change = np.random.normal(0.001, volatility)
        
        # Calculate new price: previous price * (1 + percentage change)
        # Example: $100 * (1 + 0.01) = $101 (1% increase)
        new_price = prices[-1] * (1 + change)
        
        # Ensure price doesn't go below $1 (realistic constraint)
        # max(new_price, 1) returns new_price if >= 1, else 1
        prices.append(max(new_price, 1))
    
    # Convert list to NumPy array for easier manipulation
    return np.array(prices)

# Generate training data
prices = generate_stock_prices(days=200, start_price=100, volatility=0.02)
print(f"Generated {len(prices)} days of price data")
print(f"Price range: ${prices.min():.2f} - ${prices.max():.2f}")
print(f"First 10 prices: {prices[:10]}")
print()

# 7b.2 Prepare Sequences
print("=== 7b.2 Prepare Sequences ===")
def create_price_sequences(prices, seq_length=10):
    """Create sequences for time series prediction"""
    # Lists to store input sequences and target values
    X = []  # Input sequences (historical prices)
    y = []  # Target values (next price to predict)
    
    # Create sliding window sequences
    # For each position i, use prices[i:i+seq_length] as input
    # and prices[i+seq_length] as target
    for i in range(len(prices) - seq_length):
        # Input: sequence of seq_length consecutive prices
        # prices[i:i+seq_length] gets prices from index i to i+seq_length-1
        # Example: i=0, seq_length=10 → prices[0:10] (first 10 prices)
        X.append(prices[i:i+seq_length])
        
        # Target: the next price after the sequence
        # prices[i+seq_length] is the price at position i+seq_length
        # Example: i=0, seq_length=10 → prices[10] (11th price)
        y.append(prices[i+seq_length])
    
    # Convert to NumPy arrays
    # dtype=np.float32: 32-bit float (PyTorch compatible, uses less memory)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

seq_length = 10
X, y = create_price_sequences(prices, seq_length)

print(f"Created {len(X)} sequences of length {seq_length}")
print(f"Example: Predict day {seq_length+1} from days 1-{seq_length}")
print()

# Normalize data (important for RNNs)
# Normalization helps RNNs train faster and more stably
# Formula: normalized = (value - mean) / std
# This centers data around 0 and scales to unit variance

# Calculate mean and standard deviation of input sequences
# X.mean() calculates average across all values in X
mean = X.mean()
# X.std() calculates standard deviation across all values in X
std = X.std()

# Normalize input sequences: subtract mean, divide by std
# (X - mean) centers data around 0
# / std scales data to unit variance
X_normalized = (X - mean) / std

# Normalize target values using same mean and std
# Important: Use same normalization parameters for X and y!
# This ensures predictions are in the same scale
y_normalized = (y - mean) / std

print(f"Normalized data: mean={mean:.2f}, std={std:.2f}")
print()

# Convert to tensors
X_tensor = torch.FloatTensor(X_normalized).unsqueeze(-1)  # (samples, seq_length, features)
y_tensor = torch.FloatTensor(y_normalized).unsqueeze(-1)  # (samples, 1)

print(f"Tensor shapes: X={X_tensor.shape}, y={y_tensor.shape}")
print()

# 7b.3 Build Time Series RNN
print("=== 7b.3 Build Time Series RNN ===")
class StockPriceRNN(nn.Module):
    """RNN for stock price prediction"""
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(StockPriceRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # RNN layer
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x shape: (batch, seq_length, features)
        # Example: (32, 10, 1) = 32 samples, 10 time steps, 1 feature (price) each
        
        # RNN processes sequence step by step
        # self.rnn(x) returns:
        # - rnn_out: Output at each time step, shape: (batch, seq_length, hidden_size)
        # - hidden: Final hidden state from all layers
        rnn_out, hidden = self.rnn(x)
        
        # Use last output from the sequence
        # rnn_out[:, -1, :] gets last time step for each sample
        # [:, -1, :] means: all batches, last time step, all hidden units
        # Result shape: (batch, hidden_size)
        last_output = rnn_out[:, -1, :]  # (batch, hidden_size)
        
        # Predict next price using fully connected layer
        # self.fc converts hidden state to price prediction
        # Input: (batch, hidden_size) → Output: (batch, 1)
        prediction = self.fc(last_output)  # (batch, 1)
        return prediction

model = StockPriceRNN(input_size=1, hidden_size=64, num_layers=2)
print("Model architecture:")
print(model)
print()

# 7b.4 Training
print("=== 7b.4 Training the Stock Price Predictor ===")
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

losses = []
num_epochs = 300

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
    
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.6f}")

print("Training complete!")
print()

# 7b.5 Learning Curve
print("=== 7b.5 Learning Curve ===")
plot_learning_curve(losses, title="Stock Price Prediction Training Loss", ylabel="Loss (MSE)")

# 7b.6 Make Predictions
print("=== 7b.6 Make Predictions ===")
model.eval()
with torch.no_grad():
    # Predict on training data
    predictions_normalized = model(X_tensor[:20])
    
    # Denormalize
    predictions = predictions_normalized.squeeze().numpy() * std + mean
    actual = y[:20] * std + mean
    
    print("Sample predictions (next day price):")
    print("Day | Actual Price | Predicted Price | Error")
    print("-" * 50)
    for i in range(10):
        error = abs(actual[i] - predictions[i])
        print(f"{i+1:3d} | ${actual[i]:8.2f} | ${predictions[i]:8.2f} | ${error:.2f}")
print()

# 7b.7 Multi-step Prediction
print("=== 7b.7 Multi-step Prediction ===")
def predict_future(model, last_sequence, steps=10):
    """Predict multiple future steps"""
    model.eval()
    predictions = []
    current_seq = last_sequence.copy()
    
    with torch.no_grad():
        for _ in range(steps):
            # Normalize
            seq_normalized = (current_seq - mean) / std
            x = torch.FloatTensor(seq_normalized).unsqueeze(0).unsqueeze(-1)
            
            # Predict
            pred_normalized = model(x)
            pred = pred_normalized.item() * std + mean
            
            predictions.append(pred)
            
            # Update sequence (shift and add prediction)
            current_seq = np.append(current_seq[1:], pred)
    
    return np.array(predictions)

# Predict next 10 days
last_seq = prices[-seq_length:]
future_prices = predict_future(model, last_seq, steps=10)

print(f"Last known price: ${prices[-1]:.2f}")
print("\nPredicted future prices:")
for i, price in enumerate(future_prices, 1):
    change = price - prices[-1]
    change_pct = (change / prices[-1]) * 100
    print(f"  Day +{i}: ${price:.2f} ({change_pct:+.2f}%)")
print()

# 7b.8 Important Notes
print("=== 7b.8 Important Notes ===")
print("⚠️  Stock prediction is extremely difficult!")
print("   - Markets are influenced by many factors")
print("   - Past performance doesn't guarantee future results")
print("   - This is for educational purposes only")
print()
print("✅ Real-world improvements:")
print("   - Use LSTM or GRU (better memory)")
print("   - Add multiple features (volume, indicators)")
print("   - Use attention mechanisms")
print("   - Consider external factors (news, events)")
print()

print("🎉 Stock price prediction complete!")
print("Remember: This is a simplified example for learning!")
