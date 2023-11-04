import time
from datetime import datetime as dt
import requests as rq
import pandas as pd
import numpy as np

FILEPATH = 'scraping/'

# String to unix timestamp
def getTimestamp(strDate: str) -> int:
    dayInUnix = 86400
    timestamp = 0

    if strDate == 'today':
        timestamp = dt.timestamp(dt.now())
    else:
        timestamp = dt.strptime(strDate, '%d/%m/%Y').timestamp()

    return int(timestamp - (timestamp % dayInUnix))

# Tags of all the stocks in S&P 500
def getAllTags():
    temp_df = pd.read_csv(FILEPATH + 'SP_tags.csv')
    return temp_df['stock'].tolist()

# Convert format of date
def changeDate(date:str):
    dateElems = date.split("-")
    return dateElems[2] + "/" + dateElems[1] + "/" + dateElems[0]

# Label if closing price increased
def add_label(df):
    new_row = np.zeros(len(df))
    df["label"] = new_row
    condition = df['close'].shift(1) > df['close'] 
    df['label'] = np.where(condition, 0, 1)
    df.at[0, 'label'] = 0

# Scraping variables
startDate = getTimestamp('01/11/2018')
endDate = getTimestamp('today') # actual date or 'today'
numStocks = 40 # current max = 503

# Initial DataFrame
df = pd.DataFrame([], columns=['date', 'open', 'high', 'low', 'close', 'volume'], dtype = str)

# Scrape all the stocks
for idx, stock_tag in enumerate(getAllTags()[:numStocks]):
    print(idx + 1, ':', stock_tag)

    # URL setup & GET request
    my_url = "https://query1.finance.yahoo.com/v7/finance/download/{}?period1={}&period2={}&interval=1d&events=history".format(stock_tag, startDate, endDate)
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
    r = rq.get(url=my_url, headers=headers)

    # Get raw data from request
    data = r.text
    data = data.split("\n")
    data = [day.split(",") for day in data][1:]

    # Convert raw data to DataFrame
    temp_df = pd.DataFrame(data, columns =['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume'], dtype = str)
    
    # Drop adjusted close column and add stock tag to the rows
    temp_df = temp_df.drop(['adj_close'], axis=1)
    temp_df['name'] = stock_tag

    # Add labels for closing price
    add_label(temp_df)

    # Add rows from stock to the main DataFrame
    df = pd.concat([df, temp_df], ignore_index=True)
    

# Convert format of dates
df['date'] = df['date'].apply(changeDate)
    

# Output scraped data to csv file
df.to_csv(FILEPATH + 'output_data.csv', index=False)