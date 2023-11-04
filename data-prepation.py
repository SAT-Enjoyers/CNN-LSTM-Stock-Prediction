import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

def get_time_series(df):
    unique_names = df["Name"].unique()
    time_series_open = []
    time_series_close = []
    for name in unique_names:
        query = (df["Name"] == name)
        num_true = query.sum()
        if num_true != 1259:
            continue
        else:
            new_df_open = df.loc[query,"close"][:-4]
            new_df_close = df.loc[query,"open"][:-4]
        new_df_chunks = np.split(new_df_open,5)
        for chunk in new_df_chunks:
            time_series_open.append(chunk)    
        new_df_chunks = np.split(new_df_close,5)
        for chunk in new_df_chunks:
            time_series_close.append(chunk)    
    return time_series_open,time_series_close
    
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
time_series_open,time_series_close = get_time_series(df)
time_series_open = normalise_all(time_series_open)
print(time_series_open[0])
# Define your time series data
time_series = time_series_open[0]

# Calculate the distance matrix
def recurrence_matrix(ts, epsilon, steps=None):
    if steps is None:
        steps = len(ts)
    rec_matrix = np.zeros((steps, steps))
    for i in range(steps):
        for j in range(steps):
            # Euclidean distance is used for simplicity; other norms can be considered
            if abs(ts[i] - ts[j]) < epsilon:
                rec_matrix[i, j] = 1
    return rec_matrix

# Define the threshold epsilon
epsilon = 0.1  # This threshold will need to be adjusted based on your data

# Generate the recurrence matrix
rec_mat = recurrence_matrix(time_series, epsilon)

# Plot the recurrence plot
plt.imshow(rec_mat, cmap='binary', origin='lower')
plt.colorbar()
plt.title('Recurrence Plot')
plt.xlabel('Time Index')
plt.ylabel('Time Index')
plt.show()
# time_series_open, time_series_close = get_time_series(link)
# print(time_series_open)

# def recurrence_plot():
