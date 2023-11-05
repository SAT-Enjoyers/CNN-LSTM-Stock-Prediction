import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.pyplot as plt
from pyts.image import RecurrencePlot
from mpl_toolkits.axes_grid1 import ImageGrid

DATASET_FILEPATH = 'scraping/SP_data.csv'

def get_all_time_series(df):
    unique_names = df["name"].unique()
    ts_open, ts_high, ts_low, ts_close, ts_vol = [], [], [], [], []

    for name in unique_names[1:]:
        # Get all rows belonging to a stock
        tsRows = df[df['name'] == name]

        # Ensure the number of rows is a multiple of 10
        ts_length = tsRows.shape[0]
        adjusted_length = ts_length - ts_length % 10

        # Extract time series from rows
        df_open = normalise_time_series(tsRows['open'][:adjusted_length])
        df_high = normalise_time_series(tsRows['high'][:adjusted_length])
        df_low = normalise_time_series(tsRows['low'][:adjusted_length])
        df_close = normalise_time_series(tsRows['close'][:adjusted_length])
        df_vol = normalise_time_series(tsRows['volume'][:adjusted_length])

        # Partition each time series into chunks of 10
        partitionTimeSeries(df_open, ts_open)
        partitionTimeSeries(df_high, ts_high)
        partitionTimeSeries(df_low, ts_low)
        partitionTimeSeries(df_close, ts_close)
        partitionTimeSeries(df_vol, ts_vol)

    return [ts_open, ts_high, ts_low, ts_close, ts_vol]

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
all_time_series = get_all_time_series(df)

all_time_series = np.array(all_time_series)

np.save('data_prep_out', all_time_series)

# Get the recurrence plots for all the time series
# final_recurrence_plots = []
# is_empty = True
# for time_series in all_time_series:
#     # print('1')
#     if is_empty:
#         final_recurrence_plots = RecurrencePlot(threshold='point', percentage=20).fit_transform(time_series)
#         is_empty = False
#         continue
#     final_recurrence_plots += RecurrencePlot(threshold='point', percentage=20).fit_transform(time_series)
# final_recurrence_plots = final_recurrence_plots/5

# print(final_recurrence_plots)
# print(np.shape(final_recurrence_plots))


#Plot the 50 recurrence plots
# fig = plt.figure(figsize=(10, 5))

# grid = ImageGrid(fig, 111, nrows_ncols=(5, 10), axes_pad=0.1, share_all=True)
# for i, ax in enumerate(grid):
#     ax.imshow(final_recurrence_plots[i], origin='lower')
# grid[0].get_yaxis().set_ticks([])
# grid[0].get_xaxis().set_ticks([])

# fig.suptitle(
#     "Recurrence plots for the 50 time series in the 'GunPoint' dataset",
#     y=0.92
# )

# plt.show()