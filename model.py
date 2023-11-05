import tensorflow as tf
import numpy as np
from keras.layers import Dense, LSTM
from keras.layers import Conv1D
from keras.models import Sequential
from keras.layers import MaxPooling1D
from keras.layers import TimeDistributed
from keras.layers import Flatten
from keras.layers import Dropout


def main():
    print("h")
    model = Sequential()
    cnn(model)
    lstm(model)
    
    print('Loading...')
    fileName = np.load('data_prep_out.npy')
    labels = np.load('labels.npy')
    print(np.shape(fileName))
    file_chuncks = np.split(fileName,5,axis=1)
    train_series = []
    train_labels = []
    label_chuncks = np.split(labels,5)
    for i in range(4):
        train_series.append(file_chuncks[i])
        train_labels.append(label_chuncks[i])
    test_series = file_chuncks[-1]
    test_labels = label_chuncks[-1]
    print('Train...')
    train_series = np.reshape(np.array(train_series),(47000,5,10))
    y_train = np.array(train_labels)
    y_train = y_train.reshape(47000)
    test_series = np.reshape(test_series,(11750,5,10))
    test_labels = []
    for i in range(len(test_series)-1):
        if i%1250 == 0:
            test_labels.append(0)
            continue
        if test_series[i][1][-1] < test_series[i+1][1][-1]:
            test_labels.append(1)
        else:
            test_labels.append(0)
    test_labels.append(0)
    test_labels = np.array(test_labels)
    y_test = np.array(test_labels)
    print(np.shape(y_train),np.shape(train_series))
    model.fit(train_series, y_train,
            batch_size=64,
            epochs=10,)
    score,acc = model.evaluate(test_series,y_test)
    print(score,acc)
    
    model.save("model.keras")
    
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
    model.add(Dropout(.5))
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