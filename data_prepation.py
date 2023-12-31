import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.pyplot as plt

DATASET_FILEPATH = 'scraping/SP_data.csv'

def get_all_time_series(df):
    unique_names = df["name"].unique()
    ts_open, ts_high, ts_low, ts_close, ts_vol = [], [], [], [], []
    df_labels = []

    for name in unique_names:
        
        # Get all rows belonging to a stock
        tsRows = df[df['name'] == name]

        # Ensure the number of rows is a multiple of 10
        ts_length = tsRows.shape[0]

        actual_partitions = (ts_length - 1) // 10
        adjusted_length = actual_partitions * 10 + 1


        # Extract time series from rows
        df_open = normalise_time_series(tsRows['open'][:adjusted_length])
        df_high = normalise_time_series(tsRows['high'][:adjusted_length])
        df_low = normalise_time_series(tsRows['low'][:adjusted_length])
        df_close = normalise_time_series(tsRows['close'][:adjusted_length])
        df_vol = normalise_time_series(tsRows['volume'][:adjusted_length])

        df_close = df_close.reset_index(drop=True)

        df_labels.extend([df_close[10 * i] for i in range(1, actual_partitions + 1)])

        # Partition each time series into chunks of 10
        partitionTimeSeries(df_open[:-1], ts_open)
        partitionTimeSeries(df_high[:-1], ts_high)
        partitionTimeSeries(df_low[:-1], ts_low)
        partitionTimeSeries(df_close[:-1], ts_close)
        partitionTimeSeries(df_vol[:-1], ts_vol)   

    return ([ts_open, ts_high, ts_low, ts_close, ts_vol], df_labels)

# Receives an un-normalised unsplit time series as input
# (datapoint - average) / standard deviation
def normalise_time_series(time_series):
    s_d = np.std(time_series)
    avg = time_series.sum() / len(time_series)
    return (time_series - avg) / s_d

# Partitions a normalised time series into partitions/chunks of 10
def partitionTimeSeries(ts_unsplit, ts_split):
    # Split the time series into partitions/chunks of size 10
    partitions = np.split(ts_unsplit, ts_unsplit.shape[0] / 10)
    for partition in partitions:
        ts_split.append(partition)


df = pd.read_csv(DATASET_FILEPATH)
all_time_series, labels = get_all_time_series(df)

all_time_series = np.array(all_time_series)
print(np.shape(all_time_series))

np.save('data_prep_out', all_time_series)
np.save('data_prep_out_labels', labels)