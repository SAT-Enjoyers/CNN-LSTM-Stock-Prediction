import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dense, LSTM, TimeDistributed, Flatten, Dropout


def main():

    model = Sequential()
    cnn(model)
    lstm(model)

    # Load data
    all_series = np.load('data_prep_out.npy')
    all_labels = np.load('data_prep_out_labels.npy')

    print("Reshape series")
    print(np.shape(all_series))
    all_series = np.reshape(all_series, (len(all_series[0]), 10, 5))
    print(np.shape(all_series), "\n")

    # value of when test inputs start
    test_index = int((len(all_series)*4)//5)
    print("test_index: ", test_index, "\n")

    # splits
    train_series = all_series[:test_index]
    test_series = all_series[:-test_index]
    train_labels = all_labels[:test_index]
    test_labels = all_labels[:-test_index]

    print("Train shape: ", np.shape(train_series))
    print("Test shape: ", np.shape(test_series))

    # create labels
    test_labels, train_labels = create_all_labels(all_series, test_index)

    # reshape for training
    y_train = np.array(train_labels)
    y_train = y_train.reshape(test_index)
    test_series = np.reshape(np.array(test_series),
                             (len(all_series)-test_index, 10, 5))
    test_labels = np.array(test_labels)
    y_test = np.array(test_labels)

    # train model
    model.fit(train_series, y_train,
              batch_size=64,
              epochs=10,)

    # get resuts
    correct_cnt = 0
    results = model.predict(train_series)
    print(len(results))
    for index, result in enumerate(results):
        if y_train[index] < train_series[index][-1][3] and result < train_series[index][-1][3]:
            correct_cnt += 1
        elif y_train[index] > train_series[index][-1][3] and result > train_series[index][-1][3]:
            correct_cnt += 1
    accuracy = correct_cnt/len(results)
    print(accuracy)
    correct_cnt = 0
    results = model.predict(test_series)
    print(len(results), len(y_test), len(test_series[0][-1]))
    for index, result in enumerate(results):
        if (y_test[index] < test_series[index][-1][3]) and (result[0] < test_series[index][-1][3]):
            correct_cnt += 1
        elif (y_test[index] > test_series[index][-1][3]) and (result[0] > test_series[index][-1][3]):
            correct_cnt += 1
    accuracy = correct_cnt/len(results)
    print(accuracy)
    # for i in range(len(results)):
    #     if test_series[i][-1][3]:
    #         print(results[i],y_test[i],test_series[i][-1][3])
    # print(accuracy)
    score, mae = model.evaluate(test_series, y_test)
    print(score, mae)


def cnn(model):
    sides = 5  # Size of sides
    channels = 1  # Grayscale
    timeStep = 10  # Timestep
    input_shape = (timeStep, sides)

    # model.add(TimeDistributed(Conv1D(filters=32, kernel_size=1, strides=5, activation='tanh'), input_shape=input_shape))
    # model.add(TimeDistributed(MaxPooling1D(pool_size=1)))
    model.add(Conv1D(filters=32, kernel_size=1, strides=5, activation='tanh'))
    model.add(MaxPooling1D(pool_size=1))
# From here onwards, just CNN
# Shape should be (batch_size, timesteps, features)
# batch_size = number of samples in each batch
# timesteps = The number of time steps or sequences you want to consider. This could be the length of the time series data you want to feed into the LSTM.
# features = The number of features extracted from CNN layer


def lstm(model):

    model.add(LSTM(units=64, input_shape=(10, 32)))
    model.add(Dropout(.1))
    model.add(Dense(1, activation='tanh'))
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='mean_absolute_error',
                  optimizer=optimizer,
                  metrics=['mae'])


def create_all_labels(all_series, test_index):

    all_labels = []
    # create labels
    for i in range(0, len(all_series)-1):
        # check if stock should have been bought
        all_labels.append(all_series[i+1][0][3])
        # if all_series[i][1][-1] < all_series[i+1][1][0]:
        #     all_labels.append(1)
        # else:
        #     all_labels.append(0)
    all_labels.append(all_series[-1][0][3])
    test_labels = all_labels[:-test_index]
    train_labels = all_labels[:test_index]

    print(np.shape(test_labels), np.shape(train_labels))
    return test_labels, train_labels


if __name__ == "__main__":
    main()
