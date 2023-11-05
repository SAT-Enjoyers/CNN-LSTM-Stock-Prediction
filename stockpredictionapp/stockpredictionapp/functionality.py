import time
from datetime import datetime as dt
import requests as rq
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta

endDate = 0 # actual date or 'today'

# String to unix timestamp
def getTimestamp(date):
    dayInUnix = 86400
    timestamp = 0

    if type(date) == int:
        timestamp = dt.timestamp(dt.now() - relativedelta(years=date))
    else:
        timestamp = dt.strptime(date, '%d/%m/%Y').timestamp()

    return int(timestamp - (timestamp % dayInUnix))

# Convert format of date
def changeDate(date:str):
    dateElems = date.split("-")
    return dateElems[2] + "/" + dateElems[1] + "/" + dateElems[0]

endDate = getTimestamp(endDate)

def get_one(yearAgo, stockTag):
    startDate = getTimestamp(yearAgo)
    
    # Initial DataFrame
    df = pd.DataFrame([], columns=['date', 'open', 'high', 'low', 'close', 'volume'], dtype = str)
    
    # URL setup & GET request
    my_url = f"https://query1.finance.yahoo.com/v7/finance/download/{stockTag}?period1={startDate}&period2={endDate}&interval=1d&events=history"
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
    r = rq.get(url=my_url, headers=headers)
    

    if r.status_code == 200:
        # Get raw data from request
        data = r.text
        data = data.split("\n")
        data = [day.split(",") for day in data][1:]
        
        # Convert raw data to DataFrame
        df = pd.DataFrame(data, columns =['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume'], dtype = str)
        
        # Drop adjusted close column and add stock tag to the rows
        df = df.drop(['adj_close', 'open', 'high', 'low'], axis=1)
        
    # Convert format of dates
    df['date'] = df['date'].apply(changeDate)
    
    output = df.values.tolist()
    return output

print(get_one(1, 'A'))