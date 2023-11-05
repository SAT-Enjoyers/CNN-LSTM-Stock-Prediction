import tensorflow as tf
import numpy as np
import pandas as pd
from keras.layers import Dense, LSTM
from keras.layers import Conv1D
from keras.models import Sequential
from keras.layers import MaxPooling1D
from keras.layers import TimeDistributed
from keras.layers import Flatten
from keras.layers import Dropout
from data_prepation import get_time_series
from sklearn.preprocessing import MinMaxScaler


# Generate random data
n_samples = 30000

# Example dataset with columns: 'open', 'high', 'low', 'volume', 'close'
data = {
    'open': np.random.rand(n_samples),
    'high': np.random.rand(n_samples),
    'low': np.random.rand(n_samples),
    'volume': np.random.rand(n_samples),
    # 'close': np.random.rand(n_samples)
}

data['close'] = 2 * data['open']

df = pd.DataFrame(data)

time_steps = 10

def preprocess_data(df):
    X = df[['open', 'high', 'low', 'volume']].values
    y = df['close'].values
    return X, y

x, y = preprocess_data(df)

scaler = MinMaxScaler(feature_range=(0, 1))
x = scaler.fit_transform(x)

train_size = int(len(x) * 0.8)
x_train, x_test = x[:train_size], x[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

x = np.reshape(x, (x.shape[0], x.shape[1], 1))

#y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1], 1))
#y_test = np.reshape(y_test, (y_test.shape[0], y_test.shape[1], 1))

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


print(x.shape)

model = Sequential()
model.add(LSTM(units=64, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))#(change 10,5)
model.add(Dropout(0.5))
model.add(LSTM(units=64, return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(units=64))
model.add(Dropout(0.5))
model.add(Dense(units=1,activation='sigmoid'))

optimiser = tf.keras.optimizers.Adam(lr=0.0001)
model.compile(optimizer=optimiser, loss='mean_squared_error', metrics=['mae'])

model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))

predicted_stock_price = model.predict(x_test)
#predicted_stock_price = scaler.inverse_transform(predicted_stock_price)