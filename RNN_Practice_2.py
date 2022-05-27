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

df = pd.read_csv('TESLA_STOCK_PRICE.csv')
train_df = df.iloc[0:2373,1].values
train_df = train_df.reshape(-1,1)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_train_df = scaler.fit_transform(train_df)

X_train = []
y_train = []

for i in range(100, 2373):
    X_train.append(scaled_train_df[i-100:i,0])
    y_train.append(scaled_train_df[i,0])
    
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

model = Sequential()

model.add(LSTM(100, return_sequences=(True), input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(100, return_sequences=(True)))
model.add(Dropout(0.2))
model.add(LSTM(100, return_sequences=(True)))
model.add(Dropout(0.2))
model.add(LSTM(100, return_sequences=(True)))
model.add(Dropout(0.2))
model.add(LSTM(100, return_sequences=(True)))
model.add(Dropout(0.2))
model.add(LSTM(100, return_sequences=(True)))
model.add(Dropout(0.2))
model.add(LSTM(100, return_sequences=(True)))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dropout(0.2))

model.add(Dense(1))

model.compile(optimizer='adam', loss=('mse'))

model.fit(X_train,y_train, epochs=25, batch_size=32)


Real_Stock_Price = df.iloc[2373: , 1].values


open_df = df['Open']

inputs = open_df[len(open_df)- len(Real_Stock_Price)-100:].values
inputs = inputs.reshape(-1,1)
inputs = scaler.transform(inputs)

X_test = []
for i in range(100, 121):
    X_test.append(inputs[i-100:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

Tesla_Stock_Predictions= model.predict(X_test)
Tesla_Stock_Predictions = scaler.inverse_transform(Tesla_Stock_Predictions)

plt.plot(Real_Stock_Price, color='blue', label = 'Real Tesla Stock Price')
plt.plot(Tesla_Stock_Predictions, color='red', label = 'Predicted Tesla Stock Price')
plt.xlabel('Time')
plt.ylabel('Tesla Stock Price')
plt.legend()
plt.show()

    

    
    

