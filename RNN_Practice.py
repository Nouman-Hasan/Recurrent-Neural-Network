import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf
import plotly.express as px

sns.set_theme(color_codes=True)
sns.set_style('whitegrid')
init_notebook_mode(connected=True)
cf.go_offline()

#TASK IS TO DETERMINE THE OPEN MARKET TRENDS FOR THE FIRST MONTH OF 2017
df = pd.read_csv('Google_Stock_Price_Train.csv')
df_train = df['Open'].values

df_train = df_train.reshape(-1,1)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_df_train = scaler.fit_transform(df_train)

X_train = []
y_train = []

#Creating a data structure of 60 timestamps for 1 output
for i in range(60,1258):
    X_train.append(scaled_df_train[i-60:i, 0])
    y_train.append(scaled_df_train[i, 0])
    
X_train, y_train = np.array(X_train), np.array(y_train)

#Reshaping into 3D according to input shape of RNN
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

model = Sequential()

model.add(LSTM(units= 50, return_sequences=(True), input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units= 50, return_sequences=(True)))
model.add(Dropout(0.2))
model.add(LSTM(units= 50, return_sequences=(True)))
model.add(Dropout(0.2))
model.add(LSTM(units= 50))
model.add(Dropout(0.2))

model.add(Dense(1))

model.compile(optimizer = 'adam', loss='mse')

model.fit(X_train,y_train, epochs=100, batch_size=32)


df_test = pd.read_csv('Google_Stock_Price_Test.csv')
Real_Stock_Price = df_test['Open'].values

final_df1 = pd.concat([df['Open'], df_test['Open']], axis = 0)

inputs = final_df1[len(final_df1)-len(df_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = scaler.transform(inputs)

X_test = []
for i in range(60,80):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

trend_pred = model.predict(X_test)
trend_pred = scaler.inverse_transform(trend_pred)

plt.plot(Real_Stock_Price, color='blue', label = 'Real Google Stock Price')
plt.plot(trend_pred, color='red', label = 'Predicted Google Stock Price')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()


