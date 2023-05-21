import numpy as np
import pandas as pd
from  pandas_datareader import data as pdr

import os
import csv
import datetime as dt

import yfinance as yf


from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
#import keras as k
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,LSTM


_SAVED_MODEL_PATH = "loaded_weights"
_SAVED_PREDICTIONS_PATH = "saved_predictions.csv"

yf.pdr_override()

Y_SYMBOLS = ['META']
company = Y_SYMBOLS[0]
start = dt.datetime(2012,1,1)
stop = dt.datetime(2023,1,1)

data = pdr.get_data_yahoo(Y_SYMBOLS,start=start,end=stop)

# Prepare
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

prediction_days = 60

X_train = []
y_train = []

for x in range(prediction_days,len(scaled_data)):
    X_train.append(scaled_data[x-prediction_days:x,0])
    y_train.append(scaled_data[x,0])

X_train = np.array(X_train)
y_train = np.array(y_train)

X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))


if os.path.exists(_SAVED_MODEL_PATH):
    model = tf.keras.models.load_model(_SAVED_MODEL_PATH)
else:
    model = Sequential()
    model.add(LSTM(units =50,return_sequences = True,input_shape = (X_train.shape[1],1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50,return_sequences = True))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 50))
    model.add(Dropout(0.2))
    model.add(Dense(units = 1))

    model.compile(optimizer = 'adam',loss = 'mean_squared_error')
    # Train model
    model.fit(X_train,y_train,epochs = 25,batch_size = 32)
    # save trained model
    model.save("./loaded_weights")

# Test Model

test_start = dt.datetime(2023,1,1)
test_end = dt.datetime.now()

test_data = pdr.DataReader(company,test_start,test_end)
actual_prices = test_data['Close'].values

total_dataset = pd.concat((data['Close'],test_data['Close']))

model_inputs = total_dataset[len(total_dataset)- len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1,1)
model_inputs = scaler.transform(model_inputs)

# Make Predictions

X_test = []

for x in range(prediction_days,len(model_inputs)):
    X_test.append(model_inputs[x-prediction_days:x,0])
    

X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))

prediction = []

if not os.path.exists(_SAVED_PREDICTIONS_PATH):
    predicted_prices = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)


    # plot
    # plt.plot(actual_prices,color = 'black',label =f"Actual {company}Price")
    # plt.plot(predicted_prices,color = 'green',label = f"Actual {company} Price")
    # plt.title(f"{company} Share Price")
    # plt.xlabel('Time')
    # plt.ylabel(f"{company} Share Price")
    # plt.legend()
    # plt.show()

    # predict Next Day

    real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs+1),0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data,(real_data.shape[0],real_data.shape[1],1))

    prediction = model.predict(real_data)
    prediction = scaler.inverse_transform(prediction)

    with open(_SAVED_PREDICTIONS_PATH, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Prediction'])
        writer.writerows(prediction)
else:
    with open(_SAVED_PREDICTIONS_PATH, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            prediction.append(float(row[0]))

def predict():
    return prediction[0][0]

def seven_day_data(comp):
    start = '2023-05-13'
    stop = '2023-05-20'

    df = yf.Download(comp,start = start,end = stop)
    return df['Close'].values.tolist()