# Project 5: Time Series Predictor

> **Build RNN/LSTM models for time series prediction**

**Difficulty**: ⭐⭐⭐ Advanced  
**Time**: 3-4 hours  
**Prerequisites**: Steps 0-7 (Especially 7b: Stock Prices, 7c: LSTM/GRU, 7b Advanced: Advanced Time Series)

---

## 📚 Table of Contents

1. [Project Overview](#project-overview)
2. [Problem: Time Series Forecasting](#problem-time-series-forecasting)
3. [Key Concepts](#key-concepts)
4. [Step-by-Step Guide](#step-by-step-guide)
5. [Expected Results](#expected-results)
6. [Extension Ideas](#extension-ideas)
7. [Troubleshooting](#troubleshooting)

---

## 🎯 Project Overview

This project teaches you to build **Recurrent Neural Networks (RNNs)** and **LSTMs** for time series prediction. You'll learn to:

- Preprocess time series data
- Build sequence-to-sequence models
- Handle temporal dependencies
- Make multi-step predictions

### Why Time Series?

- **Real-world applications**: Stock prices, weather, sales, energy demand
- **Temporal understanding**: Learn how models handle sequences
- **Foundation for forecasting**: Prepares you for advanced forecasting tasks

---

## 📋 Problem: Time Series Forecasting

### Task

Build models to predict future values in time series data. Choose one of these problems:

1. **Stock Price Forecasting** - Predict next day's stock price
2. **Weather Prediction** - Forecast temperature/weather
3. **Sales Forecasting** - Predict future sales

### Learning Objectives

- Understand time series preprocessing
- Build RNN/LSTM models
- Handle sequences and temporal dependencies
- Evaluate forecasting performance
- Make multi-step predictions

### Dataset Description

**Stock Price Data** (Example):
- Daily closing prices
- Features: Price, Volume, Moving averages
- Target: Next day's price

**Weather Data** (Example):
- Daily temperature
- Features: Temperature, Humidity, Pressure
- Target: Next day's temperature

---

## 🧠 Key Concepts

### 1. Time Series Characteristics

**Components**:
- **Trend**: Long-term direction (upward/downward)
- **Seasonality**: Repeating patterns (daily, weekly, yearly)
- **Cyclical**: Irregular cycles
- **Noise**: Random fluctuations

**Example**:
```
Time Series = Trend + Seasonality + Cyclical + Noise
```

### 2. Sequence Preprocessing

**Sliding Window**:
- Use past N values to predict next value
- Example: Use last 10 days to predict day 11

**Normalization**:
- Important for RNNs
- Normalize to mean=0, std=1
- Use same normalization for train/test

### 3. RNN vs LSTM

**RNN**:
- Simple, fast
- Struggles with long sequences
- Vanishing gradient problem

**LSTM**:
- Better memory
- Handles long sequences
- Solves vanishing gradient
- More parameters

---

## 🚀 Step-by-Step Guide

### Step 1: Create/Load Time Series Data

```python
import numpy as np

def generate_stock_prices(days=200, start_price=100, volatility=0.02):
    """Generate synthetic stock price data"""
    prices = [start_price]
    
    for _ in range(days - 1):
        # Random walk with slight upward trend
        change = np.random.normal(0.001, volatility)
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 1))  # Price can't go below 1
    
    return np.array(prices)

# Generate data
prices = generate_stock_prices(days=200, start_price=100, volatility=0.02)
print(f"Generated {len(prices)} days of price data")
print(f"Price range: ${prices.min():.2f} - ${prices.max():.2f}")
```

### Step 2: Create Sequences

```python
def create_sequences(data, seq_length=10):
    """Create sequences for time series prediction"""
    X = []
    y = []
    
    for i in range(len(data) - seq_length):
        # Input: sequence of past values
        X.append(data[i:i+seq_length])
        # Target: next value
        y.append(data[i+seq_length])
    
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# Create sequences
seq_length = 10
X, y = create_sequences(prices, seq_length)

print(f"Created {len(X)} sequences")
print(f"X shape: {X.shape}")  # (samples, seq_length)
print(f"y shape: {y.shape}")   # (samples,)
```

### Step 3: Normalize Data

```python
# Normalize (important for RNNs)
mean = X.mean()
std = X.std()
X_normalized = (X - mean) / std
y_normalized = (y - mean) / std

print(f"Normalized: mean={mean:.2f}, std={std:.2f}")
```

### Step 4: Build RNN Model

```python
import torch
import torch.nn as nn

class StockPriceRNN(nn.Module):
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
        rnn_out, hidden = self.rnn(x)
        
        # Use last output
        last_output = rnn_out[:, -1, :]  # (batch, hidden_size)
        
        # Predict next price
        prediction = self.fc(last_output)  # (batch, 1)
        return prediction

# Prepare data for PyTorch
X_tensor = torch.FloatTensor(X_normalized).unsqueeze(-1)  # (samples, seq_length, 1)
y_tensor = torch.FloatTensor(y_normalized).unsqueeze(-1)  # (samples, 1)
```

### Step 5: Build LSTM Model (Better for Long Sequences)

```python
class StockPriceLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(StockPriceLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer (better memory)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x shape: (batch, seq_length, features)
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use last output
        last_output = lstm_out[:, -1, :]  # (batch, hidden_size)
        
        # Predict next price
        prediction = self.fc(last_output)  # (batch, 1)
        return prediction
```

### Step 6: Train Model

```python
# Choose model (RNN or LSTM)
model = StockPriceLSTM(input_size=1, hidden_size=64, num_layers=2, output_size=1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 100
batch_size = 32

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    
    # Train in batches
    for i in range(0, len(X_tensor), batch_size):
        batch_X = X_tensor[i:i+batch_size]
        batch_y = y_tensor[i:i+batch_size]
        
        # Forward pass
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(X_tensor)*batch_size:.4f}")
```

### Step 7: Make Predictions

```python
model.eval()
with torch.no_grad():
    predictions = model(X_tensor)
    
    # Denormalize
    predictions_actual = predictions.numpy() * std + mean
    y_actual = y_tensor.numpy() * std + mean
    
    # Calculate metrics
    mse = np.mean((predictions_actual - y_actual) ** 2)
    mae = np.mean(np.abs(predictions_actual - y_actual))
    
    print(f"MSE: {mse:.2f}")
    print(f"MAE: {mae:.2f}")
```

### Step 8: Multi-Step Prediction

```python
def predict_future(model, last_sequence, steps=10):
    """Predict multiple future steps"""
    model.eval()
    predictions = []
    current_seq = last_sequence.copy()
    
    with torch.no_grad():
        for _ in range(steps):
            # Prepare input
            input_seq = torch.FloatTensor(current_seq).unsqueeze(0).unsqueeze(-1)
            
            # Predict next value
            next_pred = model(input_seq)
            next_value = next_pred.item()
            
            predictions.append(next_value)
            
            # Update sequence (sliding window)
            current_seq = np.append(current_seq[1:], next_value)
    
    return np.array(predictions)

# Predict next 10 days
last_seq = X_normalized[-1]  # Last sequence
future_predictions = predict_future(model, last_seq, steps=10)
future_predictions_actual = future_predictions * std + mean

print(f"Next 10 days predictions: {future_predictions_actual}")
```

---

## 📊 Expected Results

### Training Output

```
Creating Time Series Data...
Generated 200 days of price data
Price range: $85.23 - $115.67

Creating Sequences...
Created 190 sequences of length 10

Training LSTM Model...
Epoch 20/100, Loss: 0.1234
Epoch 40/100, Loss: 0.0876
Epoch 60/100, Loss: 0.0543
Epoch 80/100, Loss: 0.0321
Epoch 100/100, Loss: 0.0198

Evaluation:
MSE: 2.34
MAE: 1.23
R² Score: 0.89

Multi-Step Predictions (next 10 days):
[102.34, 103.12, 104.56, 105.23, 106.78, 107.45, 108.12, 109.34, 110.56, 111.78]
```

### Visualization

- **Actual vs Predicted**: Line plot showing model predictions
- **Residuals**: Error distribution
- **Multi-step Forecast**: Future predictions

---

## 💡 Extension Ideas

### Beginner Extensions

1. **Try Different Sequence Lengths**
   - Compare seq_length = 5, 10, 20, 30
   - Observe impact on prediction quality

2. **Compare RNN vs LSTM**
   - Train both models
   - Compare performance
   - Understand when to use each

3. **Experiment with Hyperparameters**
   - Hidden size: 32, 64, 128
   - Number of layers: 1, 2, 3
   - Learning rate: 0.0001, 0.001, 0.01

### Intermediate Extensions

4. **Add More Features**
   - Volume, moving averages
   - Technical indicators
   - External factors

5. **Handle Seasonality**
   - Decompose time series
   - Remove seasonality
   - Predict seasonally adjusted values

6. **Multi-Step Prediction**
   - Predict multiple steps ahead
   - Compare direct vs recursive prediction
   - Handle prediction uncertainty

### Advanced Extensions

7. **Use Real Datasets**
   - Stock market data (Yahoo Finance)
   - Weather data (NOAA)
   - Sales data

8. **Advanced Models**
   - GRU (simpler than LSTM)
   - Transformer for time series
   - Attention mechanisms

9. **Time Series Decomposition**
   - Separate trend, seasonality, noise
   - Predict each component
   - Recombine predictions

---

## 🔧 Troubleshooting

### Common Issues

**Issue 1: Predictions are flat (no variation)**
- **Solution**: Check normalization
- **Solution**: Increase model capacity
- **Solution**: Try different sequence lengths

**Issue 2: Model doesn't learn patterns**
- **Solution**: Verify data preprocessing
- **Solution**: Check gradient flow
- **Solution**: Increase training epochs

**Issue 3: Poor multi-step predictions**
- **Solution**: Use direct prediction (not recursive)
- **Solution**: Train on multi-step targets
- **Solution**: Use ensemble methods

**Issue 4: Overfitting**
- **Solution**: Add dropout
- **Solution**: Use regularization
- **Solution**: Reduce model complexity

---

## ✅ Success Criteria

- ✅ Model achieves reasonable forecasting accuracy
- ✅ Predictions show learned patterns
- ✅ Multi-step predictions are reasonable
- ✅ Code handles sequences correctly
- ✅ Visualizations are clear

---

## 🎓 Learning Outcomes

By completing this project, you will:

- ✅ Understand time series preprocessing
- ✅ Build RNN/LSTM models
- ✅ Handle temporal dependencies
- ✅ Make single and multi-step predictions
- ✅ Evaluate forecasting performance
- ✅ Understand when to use RNN vs LSTM

---

## 📖 Additional Resources

- **Step 7b Documentation**: `docs/Step_7b_Stock_Prices.md`
- **Step 7c Documentation**: `docs/Step_7c_LSTM_GRU.md`
- **Step 7b Advanced Documentation**: `docs/Step_7b_Advanced_Time_Series.md`

---

## 🔍 Real-World Applications

### Stock Price Prediction
- **Challenge**: Highly volatile, many factors
- **Approach**: Use multiple features, ensemble models
- **Limitation**: Cannot predict unexpected events

### Weather Forecasting
- **Challenge**: Complex patterns, multiple variables
- **Approach**: Use LSTM with multiple features
- **Success**: Short-term forecasts are reliable

### Sales Forecasting
- **Challenge**: Seasonality, trends, promotions
- **Approach**: Decompose time series, handle seasonality
- **Success**: Helps with inventory planning

---

**Ready to predict the future? Let's build time series models!** 🚀

**Next Steps**: After completing this project, move on to **Project 6: Complete AI Application** to build an end-to-end system.
