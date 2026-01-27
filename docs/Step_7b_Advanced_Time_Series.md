# Step 7b Advanced: Advanced Time Series Analysis

> **Master advanced time series techniques: ARIMA, seasonality, multiple series, anomaly detection**

**Time**: ~90 minutes  
**Prerequisites**: Step 7b (Stock Price Prediction)

---

## 🎯 Learning Objectives

By the end of this step, you'll understand:
- Time series components (trend, seasonality, cyclical, noise)
- ARIMA models and how they work
- Seasonality detection and removal
- Multiple time series analysis
- Forecast evaluation metrics
- Anomaly detection in time series

---

## 📚 Time Series Components

### Decomposition

Time series can be decomposed into components:

**Time Series = Trend + Seasonality + Cyclical + Noise**

### Components Explained

#### 1. Trend

**What**: Long-term direction

**Types**:
- **Upward**: Increasing over time
- **Downward**: Decreasing over time
- **Stable**: No clear direction

**Example**: Stock prices generally increasing over years

#### 2. Seasonality

**What**: Repeating patterns over fixed periods

**Common Periods**:
- **Daily**: Patterns repeat every day
- **Weekly**: Patterns repeat every week
- **Monthly**: Patterns repeat every month
- **Yearly**: Patterns repeat every year

**Example**: Ice cream sales higher in summer

#### 3. Cyclical

**What**: Irregular cycles (not fixed period)

**Example**: Business cycles (boom and bust)

#### 4. Noise

**What**: Random fluctuations

**Example**: Daily random variations

### Why Decompose?

- **Understand patterns**: See what drives the series
- **Remove components**: Isolate specific patterns
- **Better modeling**: Model components separately
- **Forecasting**: Predict each component

---

## 📈 ARIMA Models

### What is ARIMA?

**ARIMA** = AutoRegressive Integrated Moving Average

A traditional statistical approach to time series forecasting.

### Components

#### AR (AutoRegressive)

**What**: Uses past values to predict future

**Formula**: `y(t) = c + φ₁y(t-1) + φ₂y(t-2) + ... + ε(t)`

**Example**: Today's price depends on yesterday's price

**Order (p)**: How many past values to use

#### I (Integrated)

**What**: Differencing to make series stationary

**Formula**: `Δy(t) = y(t) - y(t-1)`

**Why**: Many models require stationary data (constant mean, variance)

**Order (d)**: How many times to difference

#### MA (Moving Average)

**What**: Uses past forecast errors

**Formula**: `y(t) = c + ε(t) + θ₁ε(t-1) + θ₂ε(t-2) + ...`

**Example**: If we over-predicted yesterday, adjust today

**Order (q)**: How many past errors to use

### ARIMA(p, d, q)

**Parameters**:
- **p**: AR order (past values)
- **d**: Differencing order
- **q**: MA order (past errors)

**Example**: ARIMA(2, 1, 1)
- Uses 2 past values (AR)
- Differences once (I)
- Uses 1 past error (MA)

### When to Use ARIMA

✅ **Good for**:
- Stationary time series
- Univariate (single series)
- Short-term forecasts
- Linear relationships

❌ **Not good for**:
- Non-stationary (without differencing)
- Non-linear patterns
- Long-term forecasts
- Multiple series

---

## 🔄 Seasonality Handling

### Detecting Seasonality

#### 1. Visual Inspection

**Look for**:
- Repeating patterns
- Regular cycles
- Periodic spikes

#### 2. Autocorrelation Function (ACF)

**What**: Correlation between series and lagged version

**How**: Calculate correlation at different lags

**Peaks**: Indicate seasonal periods

**Example**: Peak at lag 7 = weekly seasonality

#### 3. Fourier Analysis

**What**: Decompose into frequency components

**How**: Find dominant frequencies

**Example**: Strong 365-day frequency = yearly seasonality

### Removing Seasonality

#### 1. Differencing

**What**: Subtract value from same period last season

**Example**: `y(t) - y(t-365)` for yearly seasonality

#### 2. Moving Average

**What**: Calculate seasonal average, subtract

**Example**: Average of same day of week, subtract from each

#### 3. Seasonal Decomposition

**What**: Separate seasonal component

**Methods**: STL, X-13ARIMA-SEATS

### Why Remove Seasonality?

- **Better forecasting**: Focus on trend
- **Model simplicity**: Easier to model
- **Comparison**: Compare across seasons
- **Anomaly detection**: Find unusual patterns

---

## 📊 Multiple Time Series

### Why Multiple Series?

Many problems involve related time series:
- **Stock prices**: Multiple companies
- **Weather**: Multiple locations
- **Sales**: Multiple products
- **Economic indicators**: GDP, unemployment, inflation

### Challenges

- **Correlations**: Series may be related
- **Dependencies**: One series affects another
- **Scale differences**: Different magnitudes
- **Complexity**: More data to model

### Approaches

#### 1. Univariate Models

**What**: Model each series separately

**Pros**: Simple, independent

**Cons**: Ignores relationships

#### 2. Multivariate Models

**What**: Model all series together

**Methods**:
- **VAR (Vector AutoRegression)**: AR for multiple series
- **Multivariate LSTM**: Uses all series as input
- **Attention mechanisms**: Focus on relevant series

**Pros**: Captures relationships

**Cons**: More complex

### Correlation Analysis

**Measure relationships**:
- **Correlation matrix**: Pairwise correlations
- **Cross-correlation**: Lagged relationships
- **Granger causality**: Does one predict another?

---

## 📏 Forecast Evaluation Metrics

### Why Special Metrics?

Time series forecasts need different evaluation than classification.

### Key Metrics

#### 1. MAE (Mean Absolute Error)

**Formula**: `MAE = mean(|actual - predicted|)`

**Interpretation**: Average error magnitude

**Pros**: Easy to understand, robust to outliers

**Cons**: Doesn't penalize large errors more

#### 2. RMSE (Root Mean Squared Error)

**Formula**: `RMSE = sqrt(mean((actual - predicted)²))`

**Interpretation**: Error in same units, penalizes large errors

**Pros**: Penalizes large errors

**Cons**: Sensitive to outliers

#### 3. MAPE (Mean Absolute Percentage Error)

**Formula**: `MAPE = mean(|(actual - predicted) / actual|) * 100%`

**Interpretation**: Percentage error

**Pros**: Scale-independent, easy to interpret

**Cons**: Problems when actual ≈ 0

#### 4. MASE (Mean Absolute Scaled Error)

**Formula**: `MASE = MAE / MAE_naive`

**Interpretation**: Relative to naive forecast

**Values**:
- **MASE < 1**: Better than naive
- **MASE = 1**: Same as naive
- **MASE > 1**: Worse than naive

**Pros**: Scale-independent, comparable

#### 5. Directional Accuracy (DA)

**Formula**: `DA = mean(actual_direction == predicted_direction) * 100%`

**Interpretation**: Percentage of correct direction predictions

**When useful**: Direction matters more than magnitude

### Choosing Metrics

- **MAE/RMSE**: General purpose
- **MAPE**: Percentage errors important
- **MASE**: Compare to baseline
- **DA**: Direction critical

---

## 🚨 Anomaly Detection

### What are Anomalies?

**Anomalies** are unusual patterns or outliers in time series.

### Types of Anomalies

#### 1. Point Anomalies

**What**: Single unusual value

**Example**: Sudden spike in temperature

#### 2. Contextual Anomalies

**What**: Unusual in specific context

**Example**: High temperature in winter (normal in summer)

#### 3. Collective Anomalies

**What**: Unusual sequence

**Example**: Series of unusual values

### Detection Methods

#### 1. Statistical Methods

**Z-Score**:
- Calculate mean and std
- Flag values beyond threshold (e.g., |z| > 3)

**Pros**: Simple, fast

**Cons**: Assumes normal distribution

#### 2. Moving Average Deviation

**What**: Compare to moving average

**How**: Flag large deviations

**Pros**: Adapts to trends

**Cons**: May miss gradual changes

#### 3. LSTM Autoencoders

**What**: Learn normal patterns, detect deviations

**How**:
1. Train autoencoder on normal data
2. High reconstruction error = anomaly

**Pros**: Learns complex patterns

**Cons**: Requires training data

#### 4. Isolation Forest

**What**: Tree-based anomaly detection

**How**: Isolate anomalies in feature space

**Pros**: Handles high dimensions

**Cons**: May flag normal but rare values

#### 5. One-Class SVM

**What**: Learn boundary of normal data

**How**: Points outside boundary = anomalies

**Pros**: Good for high dimensions

**Cons**: Sensitive to parameters

### Challenges

- **False positives**: Normal but rare values
- **False negatives**: Missed anomalies
- **Context**: What's normal depends on context
- **Labeling**: Hard to get labeled anomaly data

---

## 💻 Code Examples

### AR Model

```python
def simple_ar_model(data, order=2):
    """Simple AutoRegressive model"""
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
```

### Seasonality Removal

```python
def remove_seasonality(data, period=7):
    """Remove seasonality using moving average"""
    # Calculate seasonal indices
    seasonal_indices = np.zeros(period)
    for i in range(period):
        values = [data[j * period + i] for j in range(len(data) // period)]
        seasonal_indices[i] = np.mean(values)
    
    # Remove seasonal component
    deseasonalized = data.copy()
    for i in range(len(data)):
        seasonal_idx = i % period
        deseasonalized[i] -= seasonal_indices[seasonal_idx] - np.mean(seasonal_indices)
    
    return deseasonalized
```

### Anomaly Detection

```python
def detect_anomalies_zscore(data, threshold=3):
    """Detect anomalies using Z-score"""
    mean = np.mean(data)
    std = np.std(data)
    z_scores = np.abs((data - mean) / std)
    anomalies = z_scores > threshold
    return anomalies
```

---

## 📊 Visualizations

The step includes:
1. **Component Decomposition** - Trend, seasonality, cyclical, noise
2. **AR Predictions** - Historical and forecast
3. **Seasonality Detection** - Autocorrelation, patterns
4. **Multiple Series** - Correlation analysis
5. **Forecast Evaluation** - Actual vs predicted, error distribution
6. **Anomaly Detection** - Z-score, moving average methods

---

## ✅ Key Takeaways

1. **Understand components first** - Decompose before modeling
2. **ARIMA good for stationary series** - Traditional but effective
3. **Handle seasonality explicitly** - Detect and remove
4. **Use appropriate metrics** - MAE, RMSE, MAPE, MASE, DA
5. **Anomaly detection needs context** - What's normal depends on domain

---

## 🚀 Next Steps

After this step, you can:
- Decompose time series into components
- Apply ARIMA models
- Handle seasonality
- Analyze multiple series
- Evaluate forecasts properly
- Detect anomalies

**To dive deeper**:
- Try SARIMA (seasonal ARIMA)
- Explore VAR models
- Implement LSTM autoencoders for anomalies
- Study advanced decomposition methods

---

## 📚 Additional Resources

- [ARIMA Models](https://otexts.com/fpp2/arima.html) - Comprehensive guide
- [Time Series Analysis](https://www.itl.nist.gov/div898/handbook/pmc/section4/pmc4.htm) - NIST handbook
- [Anomaly Detection Survey](https://arxiv.org/abs/2007.02500) - Recent survey

---

## 🎓 Summary

**Advanced Time Series Analysis** extends beyond basic forecasting:

1. **Components**: Understand trend, seasonality, noise
2. **ARIMA**: Traditional statistical approach
3. **Seasonality**: Detect and handle repeating patterns
4. **Multiple Series**: Model relationships between series
5. **Evaluation**: Use appropriate metrics
6. **Anomalies**: Detect unusual patterns

**Key insight**: Time series analysis requires understanding the data structure first!

---

**Happy Forecasting!** 📈🔍
