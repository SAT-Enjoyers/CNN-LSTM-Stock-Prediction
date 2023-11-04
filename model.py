import tensorflow as tf
import numpy as np
from keras.layers import Dense, LSTM
from keras.layers import Conv2D
from keras.models import Sequential
from keras.layers import MaxPooling2D
from keras.layers import TimeDistributed
from keras.layers import Flatten


def main():
    model = Sequential()
    cnn(model)
    #lstm(model)
    print(model.summary)
    

def cnn(model):
    sides = 251 # Size of sides
    channels = 1 # Grayscale
    timeStep = 10 # Timestep
    input_shape = (timeStep, sides, sides, channels)
    
    model.add(TimeDistributed(Conv2D(filters=32, kernel_size=(1,1), activation='tanh', input_shape =input_shape)))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(1, 1))))
    
    model.add(TimeDistributed(Flatten()))
    # From here onwards, just CNN


#Shape should be (batch_size, timesteps, features)
#batch_size = number of samples in each batch
#timesteps = The number of time steps or sequences you want to consider. This could be the length of the time series data you want to feed into the LSTM.
#features = The number of features extracted from CNN layer
def lstm(inputTensor, model):


    model.add(LSTM(64))
    model.add(Dense(10))
    model.add(activation = 'sigmoid')

    model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    print('Train...')
    model.fit(inputTensor, y_train,
            batch_size=64,
            epochs=10,
            validation_data=(x_test, y_test))
    score, acc = model.evaluate(x_test, y_test, batch_size=64)
    print('Test score:', score)
    print('Test accuracy:', acc)
    # lstm = tf.keras.layers.LSTM(64)
    # output = lstm(inputTensor)
    # model.add(LSTM(64))
    # model.add(Dense(10, activation='sigmoid'))

    # # Compile the model
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Make predictions with the model
    # predictions = model.predict(inputTensor[:3])
    # print(f'Predictions: {predictions}')

