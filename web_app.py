from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
import streamlit as st
import yfinance as yf


st.title("KNOW THE TREND")

user_input = st.text_input("Enter the stock watch", "BANKBARODA.NS")
start_date = '2000-01-01'
end_date = '2020-01-01'

# Fetch historical data
df = yf.download(user_input, start=start_date, end=end_date)
df.head()
st.subheader("Data from 2000 - 2020")
st.write(df.describe())


st.subheader("Closing Price VS Time Chart")
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader("Closing Price VS Time Chart with 100MA")
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader("Closing Price VS Time Chart with 100MA and 200MA")
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100, 'r', label='MA100')
plt.plot(ma200, 'g', label='MA200')
plt.plot(df.Close, 'b', label="Closing Price")


plt.legend()
st.pyplot(fig)


data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])


scaler = MinMaxScaler(feature_range=(0, 1))

data_training_array = scaler.fit_transform(data_training)


# Loading LSTM model
model = load_model("keras_model.h5")
# Testing

past_100days = data_training.tail(100)
final_df = pd.concat([past_100days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

y_predicted = model.predict(x_test)

scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted*scale_factor
y_test = y_test*scale_factor

# Final graph
st.subheader("Future Trend vs Original Trend")
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label='original price')
plt.plot(y_predicted, 'r', label='predicted price')
plt.xlabel('time')
plt.ylabel('price')
plt.legend()
st.pyplot(fig2)
