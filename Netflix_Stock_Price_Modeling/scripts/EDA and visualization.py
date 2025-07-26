import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Optional: Display charts inline (if using Jupyter Notebook)
# %matplotlib inline

# Load the dataset (if not already loaded)
file_path = r"C:\Users\oport\OneDrive\Documents\NSPP\NFLX.csv"
df = pd.read_csv(file_path, parse_dates=['Date'])
df.sort_values('Date', inplace=True)
df.set_index('Date', inplace=True)

# 1. Line plot: Closing price
plt.figure(figsize=(14, 5))
plt.plot(df['Close'], label='Close Price')
plt.title('📈 Netflix Closing Price Over Time')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# 2. Moving averages
df['MA7'] = df['Close'].rolling(window=7).mean()
df['MA30'] = df['Close'].rolling(window=30).mean()

plt.figure(figsize=(14, 5))
plt.plot(df['Close'], label='Close Price', alpha=0.5)
plt.plot(df['MA7'], label='7-Day MA', color='orange')
plt.plot(df['MA30'], label='30-Day MA', color='green')
plt.title('📉 Moving Averages of Netflix Stock')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 3. Daily returns
df['Daily Return'] = df['Close'].pct_change()

plt.figure(figsize=(14, 4))
df['Daily Return'].plot()
plt.title('🔁 Daily Return of Netflix')
plt.xlabel('Date')
plt.ylabel('Return (%)')
plt.grid(True)
plt.tight_layout()
plt.show()

# 4. Histogram of daily returns
plt.figure(figsize=(8, 4))
df['Daily Return'].hist(bins=50, edgecolor='black')
plt.title('📊 Distribution of Daily Returns')
plt.xlabel('Daily Return')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# 5. Volume over time
plt.figure(figsize=(14, 5))
plt.plot(df['Volume'], color='purple')
plt.title('📦 Trading Volume Over Time')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.grid(True)
plt.tight_layout()
plt.show()

# 6. Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('🔍 Correlation Heatmap')
plt.tight_layout()
plt.show()
