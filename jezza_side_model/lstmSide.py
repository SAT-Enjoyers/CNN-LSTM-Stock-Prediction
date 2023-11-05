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
#from data_prepation import get_time_series
from sklearn.preprocessing import MinMaxScaler


# Generate random data
# n_samples = 30000

# # Example dataset with columns: 'open', 'high', 'low', 'volume', 'close'
# data = {
#     'open': np.random.rand(n_samples),
#     'high': np.random.rand(n_samples),
#     'low': np.random.rand(n_samples),
#     'volume': np.random.rand(n_samples),
#     # 'close': np.random.rand(n_samples)
# }

# data['close'] = 2 * data['open']

# df = pd.DataFrame(data)
x = np.load('data_prep_out_X.npy')
print("X shape:", x.shape[0], ", ", x.shape[1])
y = np.load('data_prep_out_y.npy')
print("y shape: ", y.shape[0])
# def preprocess_data(df):
#     X = df[['open', 'high', 'low', 'volume']].values
#     y = df['close'].values
#     return X, y

# x, y = preprocess_data(df)
x = np.reshape(x, (x.shape[1], x.shape[0], 1))

train_size = int(len(x) * 0.8)
x_train, x_test = x[:train_size], x[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print('f', x_train.shape[0])
print('f', x_train.shape[1])
print('f', x_train.shape[2])

#y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1], 1))
#y_test = np.reshape(y_test, (y_test.shape[0], y_test.shape[1], 1))

x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
y_train = np.reshape(y_train, (y_train.shape[0], 1))
y_test = np.reshape(y_test, (y_test.shape[0], 1))

x_train_reshaped = np.reshape(x_train, (int(x_train.shape[0]/5), 5, x_train.shape[1]))
x_test = np.reshape(x_test, (int(x_test.shape[0]/5), 5, x_test.shape[1]))
y_train = np.reshape(y_train, (int(y_train.shape[0]/5), 5))
y_test = np.reshape(y_test, (int(y_test.shape[0]/5), 5))

print(x_test.shape)
print(y_test.shape)
model = Sequential()
model.add(LSTM(units=32, return_sequences=True, input_shape=(x_train.shape[0], x_train.shape[2])))#(change 10,5)
model.add(Dropout(0.2))
# model.add(LSTM(units=64, return_sequences=True))
# model.add(Dropout(0.5))
# model.add(LSTM(units=64))
# model.add(Dropout(0.5))
model.add(Dense(units=1,activation='linear'))

optimiser = tf.keras.optimizers.Adam(lr=0.001)
model.compile(optimizer=optimiser, loss='mean_squared_error', metrics=['mae'])

model.fit(x_train, y_train, epochs=2, batch_size=512)

score,acc = model.evaluate(x_test,y_test)
print(score,acc)

predicted_stock_price = model.predict(x_test)
predicted_stock_price = predicted_stock_price.flatten()
# stock_prices = []
# s_d = np.std(predicted_stock_price)
# avg = predicted_stock_price.sum() / len(predicted_stock_price)
# for z_score in predicted_stock_price:
#     stock_prices.append(z_score * s_d + avg)
# print(stock_prices)