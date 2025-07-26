import pandas as pd
import numpy as np

# Reload data if needed (optional)
file_path = r"C:\Users\oport\OneDrive\Documents\NSPP\NFLX.csv"
df = pd.read_csv(file_path, parse_dates=['Date'])
df.sort_values('Date', inplace=True)
df.set_index('Date', inplace=True)

# Feature 1: Daily return
df['Daily_Return'] = df['Close'].pct_change()

# Feature 2: Log return
df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))

# Feature 3: Lag features (Close price from previous days)
df['Lag_1'] = df['Close'].shift(1)
df['Lag_2'] = df['Close'].shift(2)
df['Lag_3'] = df['Close'].shift(3)

# Feature 4: Rolling statistics (moving average & std deviation)
df['MA_7'] = df['Close'].rolling(window=7).mean()
df['MA_30'] = df['Close'].rolling(window=30).mean()
df['STD_7'] = df['Close'].rolling(window=7).std()

# Feature 5: Price range (High - Low)
df['Price_Range'] = df['High'] - df['Low']

# Feature 6: Volatility (Rolling std of returns)
df['Volatility_7'] = df['Daily_Return'].rolling(window=7).std()

# Feature 7: Relative Strength Index (RSI) – optional but powerful
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

# Drop rows with NaN values due to rolling calculations
df.dropna(inplace=True)

# Final check
print("🧠 Engineered Features:")
print(df[['Close', 'Daily_Return', 'Lag_1', 'MA_7', 'STD_7', 'RSI_14']].head())
