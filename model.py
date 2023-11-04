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
    all_labels = []
    fileName = np.reshape(fileName,(58750,10,5))
    # create labels
    for i in range(0,len(fileName)-1):
        if fileName[i][1][0] < fileName[i+1][1][-1]:
            all_labels.append(1)
        else:
            all_labels.append(0)
    for i in range(len(fileName),len(fileName)-1):
        if fileName[i][1][-1] < fileName[i+1][1][-1]:
            all_labels.append(1)
        else:
            all_labels.append(0)
    all_labels.append(0)
    test_labels = all_labels[47000-1:]
    train_labels = all_labels[:-11751]
    np.random.seed(0)  
    # np.random.shuffle(fileName)
    labels = np.load('labels.npy')
    print(np.shape(fileName))
    file_chuncks = np.split(fileName,5,axis=1)
    train_series = []
    for i in range(4):
        train_series.append(file_chuncks[i])
    test_series = file_chuncks[-1]
    train_series = np.reshape(np.array(train_series),(47000,10,5))
    train_labels.append(0)
    print('Train...')
    y_train = np.array(train_labels)
    y_train = y_train.reshape(47000)
    test_series = np.reshape(test_series,(11750,10,5))
    test_labels = np.array(test_labels)
    y_test = np.array(test_labels)
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
    input_shape = (timeStep, sides, channels)
    
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
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    model.compile(loss='binary_crossentropy',
                optimizer=optimizer,
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