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
    
    model = Sequential()
    cnn(model)
    lstm(model)
    print('Loading...')

    # load data
    all_series = np.load('data_prep_out.npy')

    # reshape to add labels
    all_series = np.reshape(all_series,(len(all_series[0]),10,5))

    # value of when test inputs start
    test_index = 47000

    # set random seed for consistent results
    np.random.seed(0)  
    # np.random.shuffle(fileName)

    # splits
    series_chunks = np.split(all_series,5,axis=1)
    train_series = []

    for i in range(4):
        train_series.append(series_chunks[i])
        
    test_series = series_chunks[-1]
    train_series = np.reshape(np.array(train_series),(test_index,10,5))

    #create labels
    test_labels,train_labels = create_all_labels(all_series,test_index)

    # reshape for training
    y_train = np.array(train_labels)
    y_train = y_train.reshape(test_index)
    test_series = np.reshape(np.array(test_series),(len(all_series)-test_index,10,5))
    test_labels = np.array(test_labels)
    y_test = np.array(test_labels)

    #train model
    model.fit(train_series, y_train,
            batch_size=64,
            epochs=15,)
    
    #evaluate model
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
    # model.add(Dropout(.5))
    model.add(Dense(1,activation='sigmoid'))
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    model.compile(loss='binary_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])

def create_all_labels(all_series,test_index):
    all_labels = []
    # create labels
    #
    for i in range(0,test_index):
        # checks if series is increasing
        if all_series[i][1][0] < all_series[i+1][1][-1]:
            all_labels.append(1)
        else:
            all_labels.append(0)
    for i in range(test_index,len(all_series)-1):
        # check if stock should have been bought
        if all_series[i][1][-1] < all_series[i+1][1][0]:
            all_labels.append(1)
        else:
            all_labels.append(0)
    all_labels.append(0)
    test_labels = all_labels[test_index:]
    train_labels = all_labels[:test_index-len(all_series)]
    return test_labels, train_labels

main()

