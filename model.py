import tensorflow as tf
import numpy as np
from keras.layers import Dense, LSTM
from keras.layers import Conv2D
from keras.models import Sequential
from keras.layers import MaxPooling2D
from keras.layers import Reshape
from keras.layers import Flatten

def main():
    model = Sequential()
    

def cnn(images, model):
    sides = 251 # Size of sides
    channels = 1 # Grayscale
    timeStep = 10 # Timestep
    input_shape = (64, sides, sides, channels)
    
    model.add(Conv2D(filter=32, kernel_size=(1,1), activation='tanh', input_shape =input_shape))
    model.add(MaxPooling2D(pool_size=(1, 1)))
    
    model.add(Flatten())
    flattenedFeatures = model.layers[-1].output_shape[1]
    model.add(Reshape((timeStep, flattenedFeatures)))
    # From here onwards, just CNN

def lstm(inputs, model):
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



    # units = 64
    # kwargs = {
    #     # Add any additional keyword arguments you may need
    #     'dropout': 0.2,
    #     'recurrent_dropout': 0.2,
    #     'return_sequences': True
    # }

    # lstm = tf.keras.layers.LSTM(
    # units,
    # activation="tanh",
    # recurrent_activation="sigmoid",
    # use_bias=True,
    # kernel_initializer="glorot_uniform",
    # recurrent_initializer="orthogonal",
    # bias_initializer="zeros",
    # unit_forget_bias=True,
    # kernel_regularizer=None,
    # recurrent_regularizer=None,
    # bias_regularizer=None,
    # activity_regularizer=None,
    # kernel_constraint=None,
    # recurrent_constraint=None,
    # bias_constraint=None,
    # #dropout=0.0,
    # #recurrent_dropout=0.0,
    # #return_sequences=False,
    # return_state=False,
    # go_backwards=False,
    # stateful=False,
    # time_major=False,
    # unroll=False,
    # **kwargs
    # )

    # # Example input data
    # input_data = tf.random.normal(shape=(1, 10, 5))  # Assuming a batch size of 1, sequence length of 10, and 5 features

    # # Get the output by calling the layer on the input
    # output = lstm(input_data)

    # # Print the output
    # print("LSTM Output Shape:", output.shape)
    # print("LSTM Output Values:")
    # print(output)
