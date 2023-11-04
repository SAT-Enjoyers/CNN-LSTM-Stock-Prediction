import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.pyplot as plt
from pyts.image import RecurrencePlot
from mpl_toolkits.axes_grid1 import ImageGrid

def add_label(df):
    new_row = np.zeros(len(df))
    df["label"] = new_row
    condition = df['close'].shift(1)>df['close'] 
    df['label'] = np.where(condition, 0, 1)
    df.loc[df["date"] == "08/02/2013","label"] = 0
    df.to_csv("datasets/SP_data.csv")
def rm_null(df):
    null_row = df.loc[pd.isna(df["high"]),"close"]
    df.loc[pd.isna(df["high"]),"open"] = null_row
    df.loc[pd.isna(df["high"]),"high"] = null_row
    df.loc[pd.isna(df["high"]),"low"] = null_row
    null_row = df.loc[pd.isna(df["low"]),"close"]
    df.loc[pd.isna(df["low"]),"open"] = null_row
    df.loc[pd.isna(df["low"]),"high"] = null_row
    df.loc[pd.isna(df["low"]),"low"] = null_row
    return df
def get_time_series(df):
    unique_names = df["Name"].unique()
    time_series_open = []
    time_series_close = []
    time_series_high = []
    time_series_low = []
    time_series_vol = []
    for name in unique_names:
        query = (df["Name"] == name)
        num_true = query.sum()
        if num_true != 1259:
            continue
        new_df_open = df.loc[query,"close"][:-9]
        new_df_close = df.loc[query,"open"][:-9]
        new_df_high = df.loc[query,"high"][:-9]
        new_df_low = df.loc[query,"low"][:-9]
        new_df_vol = df.loc[query,"volume"][:-9]
        get_chunks(new_df_open,time_series_open)
        get_chunks(new_df_close,time_series_close)
        get_chunks(new_df_high,time_series_high)
        get_chunks(new_df_low,time_series_low)
        get_chunks(new_df_vol,time_series_vol)
    return [time_series_open,time_series_close,time_series_high,time_series_low,time_series_vol]

def get_chunks(new_df,time_series):
    new_df_chunks = np.split(new_df,125)
    for chunk in new_df_chunks:
        time_series.append(chunk)   
    
# (input - average) / standard deviation
def normalise(row,average,standard_deviation):
    return (row-average)/standard_deviation

def normalise_series(series):
    std = np.std(series)
    avg = series.sum()/len(series)
    normalised = normalise(series,std,avg)
    return normalised

def normalise_all(series_l):
    for series in range(len(series_l)):
        series_l[series] = normalise_series(series_l[series])
    return series_l
link = "datasets/SP_data.csv"
df = pd.read_csv(link)
# df = add_label(df)
df = rm_null(df)
all_time_series = get_time_series(df)
for time_series in range(len(all_time_series)):
    all_time_series[time_series] = np.array(normalise_all(all_time_series[time_series]))
all_time_series = np.array(all_time_series)
print(np.shape(all_time_series))


# Get the recurrence plots for all the time series
""" final_recurrence_plots = []
is_empty = True
for time_series in all_time_series:
    print('1')
    if is_empty:
        final_recurrence_plots = RecurrencePlot(threshold='point', percentage=20).fit_transform(time_series)
        is_empty = False
        continue
    final_recurrence_plots += RecurrencePlot(threshold='point', percentage=20).fit_transform(time_series)
final_recurrence_plots = final_recurrence_plots/5
labels = []
for time_series in all_time_series[1]:
    if time_series[0] > time_series[-1]:
        labels.append(1)
    else:
        labels.append(0)

print(np.shape(final_recurrence_plots))
print(np.shape(labels)) """
# #Plot the 50 recurrence plots
# fig = plt.figure(figsize=(10, 5))
# #Plot the 50 recurrence plots
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