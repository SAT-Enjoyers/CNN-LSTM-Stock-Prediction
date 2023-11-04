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

def lstm(model):
    data = np.random.random((1000, 28, 1))
    labels = np.random.randint(2, size = (1000, 1))

    model.add(LSTM(64, input_shape=(28, 1)))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    model.fit(data, labels, epochs=10, batch_size=32)

    # Evaluate the model
    loss, accuracy = model.evaluate(data, labels)
    print(f'Loss: {loss}, Accuracy: {accuracy}')

    # Make predictions with the model
    predictions = model.predict(data[:3])
    print(f'Predictions: {predictions}')

main()