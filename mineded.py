import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import warnings
from sklearn.metrics import mean_squared_error,accuracy_score,r2_score
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

st.write("""

# USD to INR price predictor

	""")

st.sidebar.header('User Input Parameters')

def user_input_parameters():
	date = st.sidebar.date_input('Date')
	return date

date = user_input_parameters()

# data=pd.read_csv('data2.csv',parse_dates=['Date'],header=0,squeeze=True)
# header=data.columns.values.tolist()
# print(header)
# mask=data.isnull()
# data[mask]=np.nan
# data=data.fillna(method='bfill')
# data['Date'] = pd.to_datetime(data['Date'])
# data.index=data['Date']
# data=data.set_index(data['Date'])
# plt.plot(data,/)
# new_data=data.iloc[:,1]
# new_data

# del data['Date']

forex_data = yf.download('USDINR=X', start='2018-01-01', end='2022-05-31')

# Set the index to a datetime object
forex_data.index = pd.to_datetime(forex_data.index)

# Display the last five rows
# forex_data

mask=forex_data.isnull()
# print(mask)
forex_data[mask]=np.nan
forex_data=forex_data.fillna(method='bfill')
forex_data=forex_data.drop(['Open','High','Low','Adj Close','Volume'],axis=1)

test=yf.download('USDINR=X', start='2022-06-01', end='2022-12-31')
# test
mask=test.isnull()
test[mask]=np.nan
test=test.fillna(method='bfill')
test=test.drop(['Open','High','Low','Adj Close','Volume'],axis=1)


# train=data[:'2022/05/31']
# # print(train)
# test=data['2022/06/01':]
# test

#split training test split

test = test.replace(np.nan, 0)

test_data=list()
train_set=forex_data.values.tolist()
test1=test.values.tolist()

# test

#arima_model
pred_data=list()
train_set=forex_data.values.tolist()
test1=test.values.tolist()
# print(train_set, test1)

for t in test1:
  ar=ARIMA(train_set,order=(5, 1, 0))
  model=ar.fit()
  # print("HI")
  forcast=model.forecast()[0].tolist()
  # print(forcast)
  pred_data.append(forcast)
  train_set.append(t)
# print(pred_data)

date=pd.to_datetime(date)
# df1 = pd.DataFrame(date,columns=['Date'])
# model.predict(date)
# ar=ARIMA(df1.iloc[:,0],order=(5,1,0))

fori=model.forecast()[0]
# fori=model.predict(start=df1, dynamic = False, typ = 'levels')

st.subheader("Predicted 1 $ to INR")
st.write(fori)

fig, ax = plt.subplots(figsize=(5, 2.7), layout='constrained')

#evaluate usinf mean square error

plt.plot(test1,color='red', label='Actual Data')
plt.plot(pred_data,color='green', label='Predicted Data')
ax.set_xlabel('No. of Days')  # Add an x-label to the axes.
ax.set_ylabel('INR Value of 1 USD')  # Add a y-label to the axes.
ax.set_title("USD/INR : Actual vs Predicted using ARIMA Model")
plt.legend()
plt.show()

st.subheader("Previous Data")
st.line_chart(test1)
st.line_chart(pred_data)


mse = mean_squared_error(test1,pred_data)
accuracy = r2_score(test1,pred_data)*100

st.subheader("Mean Squared Error")
st.write(mse)

st.subheader("Accuracy Of Model")
st.write(accuracy)
