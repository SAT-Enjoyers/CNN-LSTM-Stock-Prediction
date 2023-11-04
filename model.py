import tensorflow as tf
import numpy as np
from keras.layers import Dense, LSTM
from keras.layers import Conv1D
from keras.models import Sequential
from keras.layers import MaxPooling1D
from keras.layers import TimeDistributed
from keras.layers import Flatten

def main():
    print("h")
    model = Sequential()
    cnn(model)
    lstm(model)
    
    print('Loading...')
    fileName = np.load('data_prep_out.npy')
    labels = np.load('labels.npy')
    print('Train...')
    all_time_series = fileName
    all_time_series = np.reshape(all_time_series,(58750,5,10))
    y_train = np.array(labels)
    model.fit(all_time_series, y_train,
            batch_size=64,
            epochs=10,)

def cnn(model):
    sides = 5 # Size of sides
    channels = 1 # Grayscale
    timeStep = 10 # Timestep
    input_shape = (sides, timeStep, channels)
    
    model.add(TimeDistributed(Conv1D(filters=32, kernel_size=1, activation='tanh'), input_shape =input_shape))
    model.add(TimeDistributed(MaxPooling1D(pool_size=1)))
    model.add(TimeDistributed(Flatten()))
    
    # From here onwards, just CNN
#Shape should be (batch_size, timesteps, features)
#batch_size = number of samples in each batch
#timesteps = The number of time steps or sequences you want to consider. This could be the length of the time series data you want to feed into the LSTM.
#features = The number of features extracted from CNN layer
def lstm(model):


    model.add(LSTM(64))
    model.add(Dense(1,activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    
    #score, acc = model.evaluate(x_test, y_test, batch_size=64)
    # lstm = tf.keras.layers.LSTM(64)
    # output = lstm(inputTensor)
    # model.add(LSTM(64))
    # model.add(Dense(10, activation='sigmoid'))

    # # Compile the model
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Make predictions with the model
    # predictions = model.predict(inputTensor[:3])
    # print(f'Predictions: {predictions}')

main()