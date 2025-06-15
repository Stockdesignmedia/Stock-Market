
import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import joblib

model = joblib.load('ai_model/model.pkl')

st.title("📈 AI Powered Share Market Dashboard")

ticker = st.text_input("Enter Stock Ticker (eg: TCS.NS, INFY.NS)", "TCS.NS")
period = st.selectbox("Data Period", ['1d', '5d', '1mo', '3mo', '6mo'])

data = yf.download(ticker, period=period, interval="5m")
st.write(f"Showing data for {ticker}")

data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
data['EMA_20'] = ta.trend.EMAIndicator(data['Close'], window=20).ema_indicator()
data['MACD'] = ta.trend.MACD(data['Close']).macd()

st.line_chart(data[['Close', 'EMA_20']])
st.line_chart(data[['RSI']])

st.dataframe(data.tail(20))

latest = data.iloc[-1][['Close', 'RSI', 'EMA_20', 'MACD']].values.reshape(1, -1)

prediction = model.predict(latest)
signal = ["Sell", "Hold", "Buy"][int(prediction[0])]

st.subheader(f"🔍 AI Predicted Signal: {signal}")

buy_price = st.number_input("Buy Price ₹", value=float(data['Close'][-2]))
sell_price = st.number_input("Sell Price ₹", value=float(data['Close'][-1]))
profit = (sell_price - buy_price) * 10
st.success(f"Estimated Profit: ₹{profit:.2f}")
