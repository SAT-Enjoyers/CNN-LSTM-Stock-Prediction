import time
import requests as rq
import pandas as pd

def getTimestampRange(nDays: int) -> int:
    dayInUnix = 86400
    endDate = int(round(time.time()))
    endDate = endDate - (endDate % dayInUnix)
    
    return (endDate - dayInUnix * nDays, endDate)

range = getTimestampRange(365)
stock_tag = "AAL"
my_url = "https://query1.finance.yahoo.com/v7/finance/download/{}?period1={}&period2={}&interval=1d&events=history".format(stock_tag, range[0], range[1])

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}

r = rq.get(url=my_url, headers=headers)
data = r.text

data = data.split("\n")
data = [candle.split(",") for candle in data]
data = data[1:]

df = pd.DataFrame(data, columns =['date', 'open', 'high', 'low', 'close', 'nuke', 'volume'], dtype = str)

df = df.drop(['nuke'], axis=1)
df['name'] = stock_tag

def changeDate(date:str):
    dateElems = date.split("-")
    return dateElems[2] + "/" + dateElems[1] + "/" + dateElems[0]
df['date'] = df['date'].apply(changeDate)


df.to_excel("scraped_data/{}_data.xlsx".format(stock_tag), index=False)
