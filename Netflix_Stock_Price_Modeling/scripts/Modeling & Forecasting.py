import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load & prepare data (assumes you're continuing from the feature engineering step)
file_path = r"C:\Users\oport\OneDrive\Documents\NSPP\NFLX.csv"
df = pd.read_csv(file_path, parse_dates=['Date'])
df.sort_values('Date', inplace=True)
df.set_index('Date', inplace=True)

# Feature engineering (if not already done – otherwise skip this block)
df['Daily_Return'] = df['Close'].pct_change()
df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
df['Lag_1'] = df['Close'].shift(1)
df['Lag_2'] = df['Close'].shift(2)
df['Lag_3'] = df['Close'].shift(3)
df['MA_7'] = df['Close'].rolling(window=7).mean()
df['MA_30'] = df['Close'].rolling(window=30).mean()
df['STD_7'] = df['Close'].rolling(window=7).std()
df['Price_Range'] = df['High'] - df['Low']
df['Volatility_7'] = df['Daily_Return'].rolling(window=7).std()

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df['RSI_14'] = compute_rsi(df['Close'], window=14)

# Drop missing rows
df.dropna(inplace=True)

# Step 2: Select features and target
features = ['Lag_1', 'Lag_2', 'Lag_3', 'MA_7', 'MA_30', 'STD_7', 'Price_Range', 'Volatility_7', 'RSI_14']
target = 'Close'

X = df[features]
y = df[target]

# Step 3: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

# Step 4: Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Predictions
y_pred = model.predict(X_test)

# Step 6: Evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"📊 Evaluation Metrics:")
print(f"  RMSE: {rmse:.2f}")
print(f"  MAE: {mae:.2f}")
print(f"  R² Score: {r2:.4f}")

# Step 7: Plot actual vs predicted
plt.figure(figsize=(14, 6))
plt.plot(y_test.index, y_test, label='Actual Price')
plt.plot(y_test.index, y_pred, label='Predicted Price', linestyle='--')
plt.title('🎯 Actual vs Predicted Netflix Closing Price')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
