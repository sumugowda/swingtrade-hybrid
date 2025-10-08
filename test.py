import yfinance as yf
df = yf.download("INFY.NS", start="2020-01-01")
print(df.head())
