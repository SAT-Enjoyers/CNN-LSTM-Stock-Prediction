import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.pyplot as plt
from pyts.image import RecurrencePlot

def rm_null(df):
    null_row = df.loc[pd.isna(df["high"]),"close"]
    df.loc[pd.isna(df["high"]),"open"] = null_row
    df.loc[pd.isna(df["high"]),"high"] = null_row
    df.loc[pd.isna(df["high"]),"low"] = null_row
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
        new_df_open = df.loc[query,"close"][:-4]
        new_df_close = df.loc[query,"open"][:-4]
        new_df_high = df.loc[query,"high"][:-4]
        new_df_low = df.loc[query,"low"][:-4]
        new_df_vol = df.loc[query,"volume"][:-4]
        get_chunks(new_df_open,time_series_open)
        get_chunks(new_df_close,time_series_close)
        get_chunks(new_df_high,time_series_high)
        get_chunks(new_df_low,time_series_low)
        get_chunks(new_df_vol,time_series_vol)
    return [time_series_open,time_series_close,time_series_high,time_series_low,time_series_vol]

def get_chunks(new_df,time_series):
    new_df_chunks = np.split(new_df,5)
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
df = rm_null(df)
all_time_series = get_time_series(df)
for time_series in range(len(all_time_series)):
    all_time_series[time_series] = np.array(normalise_all(all_time_series[time_series]))
all_time_series = np.array(all_time_series)
print(np.shape(all_time_series))


# Get the recurrence plots for all the time series
final_recurrence_plots = []
is_empty = True
for time_series in all_time_series:
    if is_empty:
        final_recurrence_plots = RecurrencePlot(threshold='point', percentage=20).fit_transform(time_series)
        is_empty = False
        continue
    final_recurrence_plots += RecurrencePlot(threshold='point', percentage=20).fit_transform(time_series)
final_recurrence_plots = final_recurrence_plots/5

# Plot the 50 recurrence plots
# fig = plt.figure(figsize=(10, 5))
# Plot the 50 recurrence plots
# fig = plt.figure(figsize=(10, 5))

# grid = ImageGrid(fig, 111, nrows_ncols=(5, 10), axes_pad=0.1, share_all=True)
# for i, ax in enumerate(grid):
#     ax.imshow(X_rp[i], origin='lower')
# grid[0].get_yaxis().set_ticks([])
# grid[0].get_xaxis().set_ticks([])

# fig.suptitle(
#     "Recurrence plots for the 50 time series in the 'GunPoint' dataset",
#     y=0.92
# )

# plt.show()