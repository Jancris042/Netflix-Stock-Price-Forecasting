# 📈 Netflix Stock Price Forecasting

This project demonstrates time series forecasting on Netflix (NFLX) stock data using Python.

## 🔍 Objectives

- Perform exploratory data analysis (EDA)
- Engineer features such as rolling statistics and RSI
- Train a Linear Regression model to forecast the closing price
- Visualize the predicted vs actual stock prices

## 📁 Files

- `NFLX_Modeling_Forecast.ipynb`: Jupyter notebook with full workflow
- `requirements.txt`: Dependencies required to run the notebook
- `NFLX.csv`: Your historical stock price dataset (assumed to be locally available)

## 🧠 Feature Engineering

- Lag features: Close price from previous 1–3 days
- Moving averages: 7-day and 30-day
- Rolling standard deviation
- RSI (Relative Strength Index)
- Price range and daily volatility

## 📊 Model Used

- **Linear Regression** via `scikit-learn`

## 📐 Evaluation Metrics

- RMSE
- MAE
- R² Score

## 📌 Notes

- Make sure `NFLX.csv` exists at:  
  `C:\Users\oport\OneDrive\Documents\NSPP\NFLX.csv`
- You can customize the model or try more advanced ones like XGBoost or LSTM for better forecasting accuracy.
