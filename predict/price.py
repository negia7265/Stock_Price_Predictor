
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as data
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import streamlit as st

start = '2010-01-01'
end = '2023-01-01'

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')
df = data.DataReader(user_input, 'tiingo', start, end,
                     api_key='d4f33f5b52194149931140e2f518f230d8c8295b')

# Describing Data
st.subheader('Data from 2010-2023')
st.write(df.describe())

# Visualizations
x=df['close']
x = x.reset_index(drop=True)
st.subheader('Closing Price vs Time')
fig = plt.figure(figsize=(12, 6))
plt.plot(x)
st.pyplot(fig)


# x = df.reset_index()['close']
# ma100 = pd.Series(x.rolling(100).mean())
# ma200 = pd.Series(x.rolling(200).mean())
# plt.figure(figsize=(12, 6))
# plt.plot(x, label='Close')
# plt.plot(pd.Series(ma100), 'g', label='MA100')
# plt.plot(pd.Series(ma200), 'r', label='MA200')
# plt.legend()
# plt.show()

st.subheader('Closing Price vs Time chart with 100MA')
ma100 = pd.Series(x.rolling(100).mean())
fig = plt.figure(figsize=(12, 6))
plt.plot(pd.Series(ma100), 'g', label='MA100')
plt.plot(x)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA & 200MA')

ma200 = pd.Series(x.rolling(200).mean())
fig = plt.figure(figsize=(12, 6))
plt.plot(pd.Series(ma200), 'r', label='MA200')
plt.plot(pd.Series(ma100), 'g', label='MA100')
plt.plot(x)
st.pyplot(fig)

#Spliting Data into training and testing

data_training=pd.DataFrame(df['close'][0:int(len(df)*0.70)])
data_testing=pd.DataFrame(df['close'][int(len(df)*0.70):int(len(df))])
scaler=MinMaxScaler(feature_range=(0,1))

data_training_array=scaler.fit_transform(data_training)

#splitting data into x-train and y-train

# x_train=[]
# y_train=[]

# for i in range(100,data_training_array.shape[0]):
#   x_train.append(data_training_array[i-100:i])
#   y_train.append(data_training_array[i,0])
# x_train,y_train=np.array(x_train),np.array(y_train)

#Load my model

model=load_model('keras_prediction_model.h5')

#testing part

past_100_days=data_training.tail(100)
final_df=past_100_days.append(data_testing,ignore_index=True)
input_data=scaler.fit_transform(final_df)

x_test=[]
y_test=[]

for i in range (100,input_data.shape[0]):
  x_test.append(input_data[i-100:i])
  y_test.append(input_data[i,0])
x_test,y_test=np.array(x_test),np.array(y_test)
y_predicted=model.predict(x_test)
scaler=scaler.scale_
scale_factor=1/scaler[0]
y_predicted=y_predicted*scale_factor
y_test=y_test*scale_factor

#Final Graph
st.subheader('Predictions vs Original')
fig2=plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='Original Price')
plt.plot(y_predicted,'r',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig2)