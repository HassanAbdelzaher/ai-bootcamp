"""
Step 7b Advanced — Advanced Time Series Analysis
Goal: Learn advanced time series techniques: ARIMA, seasonality, multiple series, anomaly detection
Tools: Python + NumPy + Matplotlib + PyTorch
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("Step 7b Advanced: Advanced Time Series Analysis")
print("=" * 70)
print()
print("Goal: Master advanced time series techniques")
print()

# ============================================================================
# 7b-adv.1 Time Series Components
# ============================================================================
print("=== 7b-adv.1 Time Series Components ===")
print()
print("Time series can be decomposed into components:")
print("  • Trend: Long-term direction (upward, downward, stable)")
print("  • Seasonality: Repeating patterns (daily, weekly, yearly)")
print("  • Cyclical: Irregular cycles (business cycles)")
print("  • Noise: Random fluctuations")
print()
print("Time Series = Trend + Seasonality + Cyclical + Noise")
print()

# Generate time series with components
np.random.seed(42)
days = 365
t = np.arange(days)

# Trend component (upward)
# Linear trend: starts at 100, increases by 0.1 per day
# Formula: value = 100 + 0.1 * day_number
# This represents long-term growth (e.g., increasing sales over time)
trend = 100 + 0.1 * t

# Seasonality (yearly pattern - 365 days)
# Repeating pattern that cycles every 365 days (e.g., seasonal sales)
# np.sin(2 * np.pi * t / 365): Sine wave with period 365 days
# np.cos(2 * np.pi * t / 365): Cosine wave with period 365 days
# Combining sin and cos creates more complex seasonal patterns
# 10 * sin + 5 * cos: Different amplitudes for more realistic pattern
seasonality = 10 * np.sin(2 * np.pi * t / 365) + 5 * np.cos(2 * np.pi * t / 365)

# Cyclical (longer cycles)
# Irregular cycles longer than seasonality (e.g., business cycles)
# 2 * np.pi * t / 180: Period of 180 days (6 months)
# 3 * sin: Amplitude of 3 (smaller than seasonality)
cyclical = 3 * np.sin(2 * np.pi * t / 180)  # 6-month cycle

# Noise
# Random fluctuations (unpredictable variations)
# np.random.normal(0, 2, days): Random values from normal distribution
# mean=0: Centered around zero
# std=2: Standard deviation of 2 (controls how much variation)
# days: Number of values to generate
noise = np.random.normal(0, 2, days)

# Combined time series
# Add all components together
# Real time series = Trend + Seasonality + Cyclical + Noise
time_series = trend + seasonality + cyclical + noise

print(f"Generated {days} days of time series data")
print()

# Visualize components
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

axes[0, 0].plot(t, trend, linewidth=2, color='blue')
axes[0, 0].set_title('Trend Component', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Day')
axes[0, 0].set_ylabel('Value')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(t, seasonality, linewidth=2, color='green')
axes[0, 1].set_title('Seasonality Component', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Day')
axes[0, 1].set_ylabel('Value')
axes[0, 1].grid(True, alpha=0.3)

axes[0, 2].plot(t, cyclical, linewidth=2, color='orange')
axes[0, 2].set_title('Cyclical Component', fontsize=12, fontweight='bold')
axes[0, 2].set_xlabel('Day')
axes[0, 2].set_ylabel('Value')
axes[0, 2].grid(True, alpha=0.3)

axes[1, 0].plot(t, noise, linewidth=1, color='red', alpha=0.7)
axes[1, 0].set_title('Noise Component', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Day')
axes[1, 0].set_ylabel('Value')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(t, time_series, linewidth=2, color='purple')
axes[1, 1].set_title('Combined Time Series', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Day')
axes[1, 1].set_ylabel('Value')
axes[1, 1].grid(True, alpha=0.3)

# Decomposition (simple moving average)
# Extract trend using moving average
# window: Number of days to average over (30-day window)
window = 30

# Moving average: Average of last 'window' days
# np.ones(window): Array of ones [1, 1, 1, ..., 1] (window elements)
# / window: Divide by window to get average
# np.convolve(..., mode='same'): Convolve (sliding window average)
#   - mode='same': Output has same length as input
#   - Each point is average of surrounding window points
# This smooths out short-term fluctuations to reveal trend
ma_trend = np.convolve(time_series, np.ones(window)/window, mode='same')

# Detrended: Remove trend to see seasonality and cycles
# Subtract moving average trend from original series
# Result shows seasonal and cyclical patterns more clearly
detrended = time_series - ma_trend

axes[1, 2].plot(t, time_series, linewidth=1, color='purple', alpha=0.5, label='Original')
axes[1, 2].plot(t, ma_trend, linewidth=2, color='blue', label='Trend (MA)')
axes[1, 2].set_title('Trend Extraction', fontsize=12, fontweight='bold')
axes[1, 2].set_xlabel('Day')
axes[1, 2].set_ylabel('Value')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Understanding components helps with:")
print("  • Removing trend/seasonality for better predictions")
print("  • Identifying patterns")
print("  • Choosing appropriate models")
print()

# ============================================================================
# 7b-adv.2 ARIMA Models (Traditional Approach)
# ============================================================================
print("=== 7b-adv.2 ARIMA Models ===")
print()
print("ARIMA = AutoRegressive Integrated Moving Average")
print()
print("Components:")
print("  • AR (AutoRegressive): Uses past values")
print("    y(t) = c + φ₁y(t-1) + φ₂y(t-2) + ... + ε(t)")
print()
print("  • I (Integrated): Differencing to make stationary")
print("    Δy(t) = y(t) - y(t-1)")
print()
print("  • MA (Moving Average): Uses past forecast errors")
print("    y(t) = c + ε(t) + θ₁ε(t-1) + θ₂ε(t-2) + ...")
print()
print("ARIMA(p, d, q):")
print("  • p: AR order (how many past values)")
print("  • d: Differencing order (how many times to difference)")
print("  • q: MA order (how many past errors)")
print()

# Simple AR model implementation
def simple_ar_model(data, order=2):
    """Simple AutoRegressive model"""
    # Use linear regression to estimate AR coefficients
    X = []
    y = []
    
    for i in range(order, len(data)):
        X.append(data[i-order:i])
        y.append(data[i])
    
    X = np.array(X)
    y = np.array(y)
    
    # Solve: y = X @ coefficients
    coefficients = np.linalg.lstsq(X, y, rcond=None)[0]
    
    return coefficients

# Fit AR model
ar_coeffs = simple_ar_model(time_series[:200], order=2)
print(f"AR(2) Model Coefficients: {ar_coeffs}")
print()

# Make predictions with AR model
def ar_predict(data, coefficients, steps=10):
    """Predict using AR model"""
    predictions = []
    current_data = data[-len(coefficients):].copy()
    
    for _ in range(steps):
        pred = np.dot(coefficients, current_data)
        predictions.append(pred)
        current_data = np.append(current_data[1:], pred)
    
    return np.array(predictions)

ar_predictions = ar_predict(time_series[:200], ar_coeffs, steps=20)
actual_future = time_series[200:220]

print("AR Model Predictions:")
print(f"  Actual values: {actual_future[:5]}")
print(f"  Predicted values: {ar_predictions[:5]}")
print()

# Visualize AR predictions
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(range(200), time_series[:200], label='Historical Data', linewidth=2, color='blue')
ax.plot(range(200, 220), actual_future, label='Actual Future', linewidth=2, color='green', linestyle='--')
ax.plot(range(200, 220), ar_predictions, label='AR Predictions', linewidth=2, color='red', marker='o', markersize=6)
ax.axvline(x=200, color='black', linestyle=':', linewidth=2, label='Prediction Start')
ax.set_xlabel('Day', fontsize=12, fontweight='bold')
ax.set_ylabel('Value', fontsize=12, fontweight='bold')
ax.set_title('AR Model: Historical Data and Predictions', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================================
# 7b-adv.3 Seasonality Handling
# ============================================================================
print("=== 7b-adv.3 Seasonality Handling ===")
print()
print("Seasonality: Repeating patterns over fixed periods")
print()
print("Common Seasonalities:")
print("  • Daily: Patterns repeat every day")
print("  • Weekly: Patterns repeat every week")
print("  • Monthly: Patterns repeat every month")
print("  • Yearly: Patterns repeat every year")
print()

# Generate data with strong seasonality
np.random.seed(42)
days_with_season = 730  # 2 years
t_season = np.arange(days_with_season)

# Strong weekly seasonality (7-day pattern)
weekly_season = 5 * np.sin(2 * np.pi * t_season / 7)

# Strong yearly seasonality (365-day pattern)
yearly_season = 10 * np.sin(2 * np.pi * t_season / 365)

# Trend
trend_season = 100 + 0.05 * t_season

# Combined
seasonal_series = trend_season + weekly_season + yearly_season + np.random.normal(0, 1, days_with_season)

print("Seasonal Time Series:")
print(f"  Length: {days_with_season} days (2 years)")
print(f"  Weekly pattern: 7-day cycle")
print(f"  Yearly pattern: 365-day cycle")
print()

# Detect seasonality using autocorrelation
def autocorrelation(data, max_lag=50):
    """Calculate autocorrelation"""
    n = len(data)
    mean = np.mean(data)
    autocorr = []
    
    for lag in range(max_lag):
        if lag == 0:
            autocorr.append(1.0)
        else:
            numerator = np.sum((data[lag:] - mean) * (data[:-lag] - mean))
            denominator = np.sum((data - mean) ** 2)
            autocorr.append(numerator / denominator if denominator > 0 else 0)
    
    return np.array(autocorr)

autocorr = autocorrelation(seasonal_series, max_lag=50)

# Find peaks (seasonal periods)
peaks = []
for i in range(1, len(autocorr) - 1):
    if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1] and autocorr[i] > 0.3:
        peaks.append(i)

print(f"Detected seasonal periods (lags with high autocorrelation): {peaks[:5]}")
print("  (7 and multiples of 7 indicate weekly seasonality)")
print()

# Visualize seasonality
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Time series
axes[0, 0].plot(t_season, seasonal_series, linewidth=1, color='purple', alpha=0.7)
axes[0, 0].set_title('Time Series with Seasonality', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Day')
axes[0, 0].set_ylabel('Value')
axes[0, 0].grid(True, alpha=0.3)

# Autocorrelation
axes[0, 1].plot(range(len(autocorr)), autocorr, linewidth=2, color='steelblue')
axes[0, 1].axhline(y=0, color='black', linestyle='--', linewidth=1)
for peak in peaks[:5]:
    axes[0, 1].axvline(x=peak, color='red', linestyle=':', alpha=0.5)
    axes[0, 1].text(peak, autocorr[peak] + 0.05, f'Lag {peak}', rotation=90, fontsize=9)
axes[0, 1].set_title('Autocorrelation Function', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Lag')
axes[0, 1].set_ylabel('Autocorrelation')
axes[0, 1].grid(True, alpha=0.3)

# Weekly pattern (first 30 days)
axes[1, 0].plot(t_season[:30], seasonal_series[:30], 'o-', linewidth=2, markersize=6, color='green')
axes[1, 0].set_title('Weekly Pattern (First 30 Days)', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Day')
axes[1, 0].set_ylabel('Value')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].axvline(x=7, color='red', linestyle='--', alpha=0.5, label='Week 1')
axes[1, 0].axvline(x=14, color='red', linestyle='--', alpha=0.5, label='Week 2')
axes[1, 0].legend()

# Yearly pattern (overlay 2 years)
year1 = seasonal_series[:365]
year2 = seasonal_series[365:730]
axes[1, 1].plot(range(365), year1, linewidth=2, color='blue', alpha=0.7, label='Year 1')
axes[1, 1].plot(range(365), year2, linewidth=2, color='orange', alpha=0.7, label='Year 2')
axes[1, 1].set_title('Yearly Pattern Comparison', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Day of Year')
axes[1, 1].set_ylabel('Value')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Remove seasonality (deseasonalize)
def remove_seasonality(data, period=7):
    """Remove seasonality using moving average"""
    # Calculate seasonal indices
    num_periods = len(data) // period
    seasonal_indices = np.zeros(period)
    
    for i in range(period):
        values = []
        for j in range(num_periods):
            idx = j * period + i
            if idx < len(data):
                values.append(data[idx])
        seasonal_indices[i] = np.mean(values) if values else 0
    
    # Remove seasonal component
    deseasonalized = data.copy()
    for i in range(len(data)):
        seasonal_idx = i % period
        deseasonalized[i] -= seasonal_indices[seasonal_idx] - np.mean(seasonal_indices)
    
    return deseasonalized, seasonal_indices

deseasonalized, seasonal_idx = remove_seasonality(seasonal_series, period=7)

print("Deseasonalization:")
print(f"  Original mean: {np.mean(seasonal_series):.2f}")
print(f"  Deseasonalized mean: {np.mean(deseasonalized):.2f}")
print(f"  Seasonal indices: {seasonal_idx[:7]}")
print()

# ============================================================================
# 7b-adv.4 Multiple Time Series
# ============================================================================
print("=== 7b-adv.4 Multiple Time Series ===")
print()
print("Many problems involve multiple related time series:")
print("  • Stock prices of multiple companies")
print("  • Temperature in multiple cities")
print("  • Sales of multiple products")
print("  • Economic indicators")
print()

# Generate multiple correlated time series
np.random.seed(42)
num_series = 3
days_multi = 200

# Base trend
base_trend = 100 + 0.1 * np.arange(days_multi)

# Generate correlated series
multi_series = []
for i in range(num_series):
    # Each series has base trend + unique component + shared noise
    unique_component = (i + 1) * 5 * np.sin(2 * np.pi * np.arange(days_multi) / 50)
    shared_noise = np.random.normal(0, 2, days_multi)
    series = base_trend + unique_component + shared_noise
    multi_series.append(series)

multi_series = np.array(multi_series)

print(f"Generated {num_series} correlated time series")
print(f"  Series 1: Mean={np.mean(multi_series[0]):.2f}, Std={np.std(multi_series[0]):.2f}")
print(f"  Series 2: Mean={np.mean(multi_series[1]):.2f}, Std={np.std(multi_series[1]):.2f}")
print(f"  Series 3: Mean={np.mean(multi_series[2]):.2f}, Std={np.std(multi_series[2]):.2f}")
print()

# Calculate correlation
correlation_matrix = np.corrcoef(multi_series)
print("Correlation Matrix:")
print("  Series 1 vs Series 2:", f"{correlation_matrix[0, 1]:.3f}")
print("  Series 1 vs Series 3:", f"{correlation_matrix[0, 2]:.3f}")
print("  Series 2 vs Series 3:", f"{correlation_matrix[1, 2]:.3f}")
print()

# Visualize multiple series
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# All series together
for i in range(num_series):
    axes[0, 0].plot(range(days_multi), multi_series[i], linewidth=2, alpha=0.7, label=f'Series {i+1}')
axes[0, 0].set_title('Multiple Time Series', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Day')
axes[0, 0].set_ylabel('Value')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Correlation heatmap
im = axes[0, 1].imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
axes[0, 1].set_xticks(range(num_series))
axes[0, 1].set_yticks(range(num_series))
axes[0, 1].set_xticklabels([f'Series {i+1}' for i in range(num_series)])
axes[0, 1].set_yticklabels([f'Series {i+1}' for i in range(num_series)])
axes[0, 1].set_title('Correlation Matrix', fontsize=12, fontweight='bold')
for i in range(num_series):
    for j in range(num_series):
        text = axes[0, 1].text(j, i, f'{correlation_matrix[i, j]:.2f}',
                              ha="center", va="center", color="white", fontweight='bold')
plt.colorbar(im, ax=axes[0, 1])

# Scatter plots
axes[1, 0].scatter(multi_series[0], multi_series[1], alpha=0.6, s=50)
axes[1, 0].set_xlabel('Series 1', fontsize=11, fontweight='bold')
axes[1, 0].set_ylabel('Series 2', fontsize=11, fontweight='bold')
axes[1, 0].set_title('Series 1 vs Series 2', fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].scatter(multi_series[0], multi_series[2], alpha=0.6, s=50, color='green')
axes[1, 1].set_xlabel('Series 1', fontsize=11, fontweight='bold')
axes[1, 1].set_ylabel('Series 3', fontsize=11, fontweight='bold')
axes[1, 1].set_title('Series 1 vs Series 3', fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Multi-series Models:")
print("  • VAR (Vector AutoRegression): Predicts all series together")
print("  • Multivariate LSTM: Uses all series as input")
print("  • Attention mechanisms: Focus on relevant series")
print()

# ============================================================================
# 7b-adv.5 Time Series Forecasting Evaluation
# ============================================================================
print("=== 7b-adv.5 Time Series Forecasting Evaluation ===")
print()
print("Evaluating time series forecasts requires special metrics:")
print()

# Generate predictions and actuals for evaluation
np.random.seed(42)
actual_values = time_series[200:250]
predicted_values = time_series[200:250] + np.random.normal(0, 3, 50)  # Add some error

# Calculate evaluation metrics
def calculate_forecast_metrics(actual, predicted):
    """Calculate time series forecast metrics"""
    # Mean Absolute Error (MAE)
    mae = np.mean(np.abs(actual - predicted))
    
    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    
    # Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((actual - predicted) / (actual + 1e-9))) * 100
    
    # Mean Absolute Scaled Error (MASE)
    # Scale by naive forecast (using previous value)
    naive_forecast = np.roll(actual, 1)
    naive_forecast[0] = actual[0]
    mae_naive = np.mean(np.abs(actual - naive_forecast))
    mase = mae / (mae_naive + 1e-9)
    
    # Directional Accuracy (DA)
    actual_direction = np.diff(actual) > 0
    predicted_direction = np.diff(predicted) > 0
    da = np.mean(actual_direction == predicted_direction) * 100
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'MASE': mase,
        'DA': da
    }

metrics = calculate_forecast_metrics(actual_values, predicted_values)

print("Forecast Evaluation Metrics:")
print(f"  MAE (Mean Absolute Error): {metrics['MAE']:.3f}")
print(f"  RMSE (Root Mean Squared Error): {metrics['RMSE']:.3f}")
print(f"  MAPE (Mean Absolute % Error): {metrics['MAPE']:.2f}%")
print(f"  MASE (Mean Absolute Scaled Error): {metrics['MASE']:.3f}")
print(f"  DA (Directional Accuracy): {metrics['DA']:.1f}%")
print()
print("Interpretation:")
print("  • MAE/RMSE: Lower is better")
print("  • MAPE: Lower is better (percentage error)")
print("  • MASE < 1: Better than naive forecast")
print("  • DA: Higher is better (correct direction)")
print()

# Visualize forecast evaluation
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Actual vs Predicted
axes[0, 0].plot(actual_values, label='Actual', linewidth=2, color='blue', marker='o', markersize=4)
axes[0, 0].plot(predicted_values, label='Predicted', linewidth=2, color='red', marker='s', markersize=4, alpha=0.7)
axes[0, 0].set_title('Actual vs Predicted Values', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Time Step')
axes[0, 0].set_ylabel('Value')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Scatter plot
axes[0, 1].scatter(actual_values, predicted_values, alpha=0.6, s=60, edgecolors='black', linewidth=1)
# Perfect prediction line
min_val = min(actual_values.min(), predicted_values.min())
max_val = max(actual_values.max(), predicted_values.max())
axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
axes[0, 1].set_xlabel('Actual Values', fontsize=11, fontweight='bold')
axes[0, 1].set_ylabel('Predicted Values', fontsize=11, fontweight='bold')
axes[0, 1].set_title('Prediction Accuracy', fontsize=12, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Error distribution
errors = actual_values - predicted_values
axes[1, 0].hist(errors, bins=15, alpha=0.7, color='steelblue', edgecolor='black')
axes[1, 0].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
axes[1, 0].axvline(x=np.mean(errors), color='green', linestyle='--', linewidth=2, label=f'Mean Error ({np.mean(errors):.2f})')
axes[1, 0].set_xlabel('Error (Actual - Predicted)', fontsize=11, fontweight='bold')
axes[1, 0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
axes[1, 0].set_title('Error Distribution', fontsize=12, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Metrics bar chart
metric_names = list(metrics.keys())
metric_values = [metrics[m] for m in metric_names]
# Normalize MASE and DA for visualization
normalized_values = []
for i, (name, val) in enumerate(zip(metric_names, metric_values)):
    if name == 'MASE':
        normalized_values.append(val * 10)  # Scale for visualization
    elif name == 'DA':
        normalized_values.append(val / 10)  # Scale for visualization
    elif name == 'MAPE':
        normalized_values.append(val / 10)  # Scale for visualization
    else:
        normalized_values.append(val)

colors = ['steelblue', 'coral', 'lightgreen', 'plum', 'orange']
bars = axes[1, 1].bar(metric_names, normalized_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
axes[1, 1].set_ylabel('Normalized Score', fontsize=11, fontweight='bold')
axes[1, 1].set_title('Forecast Metrics (Normalized)', fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='y')

for bar, name, val in zip(bars, metric_names, metric_values):
    height = bar.get_height()
    if name == 'MAPE':
        label = f'{val:.1f}%'
    elif name == 'DA':
        label = f'{val:.1f}%'
    elif name == 'MASE':
        label = f'{val:.2f}'
    else:
        label = f'{val:.2f}'
    axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                   label, ha='center', va='bottom', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.show()

# ============================================================================
# 7b-adv.6 Anomaly Detection in Time Series
# ============================================================================
print("=== 7b-adv.6 Anomaly Detection in Time Series ===")
print()
print("Anomaly Detection: Find unusual patterns or outliers")
print()
print("Types of Anomalies:")
print("  • Point anomalies: Single unusual value")
print("  • Contextual anomalies: Unusual in context")
print("  • Collective anomalies: Unusual sequence")
print()

# Create time series with anomalies
np.random.seed(42)
normal_series = 100 + 0.1 * np.arange(200) + np.random.normal(0, 2, 200)

# Add anomalies
anomaly_indices = [50, 100, 150]
series_with_anomalies = normal_series.copy()
series_with_anomalies[50] += 15  # Point anomaly
series_with_anomalies[100] -= 12  # Point anomaly
series_with_anomalies[150] += 20  # Point anomaly

print(f"Time series with {len(anomaly_indices)} injected anomalies")
print()

# Simple anomaly detection methods

# Method 1: Statistical (Z-score)
def detect_anomalies_zscore(data, threshold=3):
    """Detect anomalies using Z-score"""
    mean = np.mean(data)
    std = np.std(data)
    z_scores = np.abs((data - mean) / (std + 1e-9))
    anomalies = z_scores > threshold
    return anomalies, z_scores

anomalies_zscore, z_scores = detect_anomalies_zscore(series_with_anomalies, threshold=2.5)

# Method 2: Moving average deviation
def detect_anomalies_ma(data, window=10, threshold=2):
    """Detect anomalies using moving average"""
    ma = np.convolve(data, np.ones(window)/window, mode='same')
    residuals = np.abs(data - ma)
    threshold_val = np.mean(residuals) + threshold * np.std(residuals)
    anomalies = residuals > threshold_val
    return anomalies, residuals

anomalies_ma, residuals = detect_anomalies_ma(series_with_anomalies, window=10, threshold=2.5)

# Method 3: LSTM-based (conceptual)
print("Anomaly Detection Methods:")
print("  1. Statistical (Z-score):")
print(f"     Detected {np.sum(anomalies_zscore)} anomalies")
print("  2. Moving Average Deviation:")
print(f"     Detected {np.sum(anomalies_ma)} anomalies")
print("  3. LSTM-based:")
print("     Train LSTM to predict next value")
print("     Large prediction error = anomaly")
print()

# Visualize anomaly detection
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Time series with anomalies
axes[0, 0].plot(series_with_anomalies, linewidth=2, color='blue', alpha=0.7, label='Time Series')
axes[0, 0].scatter(anomaly_indices, series_with_anomalies[anomaly_indices], 
                  s=200, color='red', marker='X', label='Injected Anomalies', zorder=5)
axes[0, 0].set_title('Time Series with Anomalies', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Time Step')
axes[0, 0].set_ylabel('Value')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Z-score method
axes[0, 1].plot(z_scores, linewidth=2, color='steelblue', label='Z-Score')
axes[0, 1].axhline(y=2.5, color='red', linestyle='--', linewidth=2, label='Threshold (2.5)')
axes[0, 1].scatter(np.where(anomalies_zscore)[0], z_scores[anomalies_zscore],
                  s=150, color='red', marker='o', label='Detected Anomalies', zorder=5)
axes[0, 1].set_title('Z-Score Anomaly Detection', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Time Step')
axes[0, 1].set_ylabel('Z-Score')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Moving average method
ma = np.convolve(series_with_anomalies, np.ones(10)/10, mode='same')
axes[1, 0].plot(series_with_anomalies, linewidth=1, color='blue', alpha=0.5, label='Original')
axes[1, 0].plot(ma, linewidth=2, color='green', label='Moving Average')
axes[1, 0].scatter(np.where(anomalies_ma)[0], series_with_anomalies[anomalies_ma],
                  s=150, color='red', marker='X', label='Detected Anomalies', zorder=5)
axes[1, 0].set_title('Moving Average Anomaly Detection', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Time Step')
axes[1, 0].set_ylabel('Value')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Residuals
axes[1, 1].plot(residuals, linewidth=2, color='orange', label='Residuals')
threshold_val = np.mean(residuals) + 2.5 * np.std(residuals)
axes[1, 1].axhline(y=threshold_val, color='red', linestyle='--', linewidth=2, label='Threshold')
axes[1, 1].scatter(np.where(anomalies_ma)[0], residuals[anomalies_ma],
                  s=150, color='red', marker='o', label='Anomalies', zorder=5)
axes[1, 1].set_title('Residuals from Moving Average', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Time Step')
axes[1, 1].set_ylabel('Residual')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Advanced Anomaly Detection:")
print("  • LSTM Autoencoders: Learn normal patterns, detect deviations")
print("  • Isolation Forest: Tree-based anomaly detection")
print("  • One-Class SVM: Learn normal boundary")
print("  • Transformer-based: Attention to unusual patterns")
print()

# ============================================================================
# 7b-adv.7 Summary
# ============================================================================
print("=== 7b-adv.7 Summary ===")
print()
print("✅ You've learned:")
print("  • Time series components (trend, seasonality, noise)")
print("  • ARIMA models (traditional approach)")
print("  • Seasonality detection and removal")
print("  • Multiple time series analysis")
print("  • Forecast evaluation metrics")
print("  • Anomaly detection methods")
print()
print("🎯 Key Takeaways:")
print("  1. Understand components before modeling")
print("  2. ARIMA good for stationary series")
print("  3. Handle seasonality explicitly")
print("  4. Use appropriate evaluation metrics")
print("  5. Anomaly detection needs domain knowledge")
print()

print("=" * 70)
print("Step 7b Advanced Complete! You understand advanced time series!")
print("=" * 70)
