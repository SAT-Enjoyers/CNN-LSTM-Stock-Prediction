import tensorflow as tf
import numpy as np
from keras.layers import Dense, LSTM,Input
from keras.layers import Conv2D
from keras.models import Sequential
from keras.layers import MaxPooling2D
from keras.layers import TimeDistributed
from keras.layers import Flatten
from keras.layers import Dropout


def main():
    model = Sequential()
    cnn(model)
    model.summary()
    data = np.load('data_prep_out_rp.npy')
def cnn(model):
    input_shape = (60,82,1)
    model.add(Input(shape = input_shape))
    model.add(Conv2D(filters=8, kernel_size=(1,82), activation='relu'))
    model.add(Conv2D(filters=8, kernel_size=(3,1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,1)))
    model.add(Conv2D(filters=8, kernel_size=(3,1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,1)))
    model.add(Flatten())
    Dropout(0.5)
    model.add(Dense(1, activation='sigmoid'))
    return model

main()