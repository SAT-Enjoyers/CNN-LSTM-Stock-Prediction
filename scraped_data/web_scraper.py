import time
import requests as rq
import pandas as pd

def getTimestampRange(nDays: int) -> int:
    dayInUnix = 86400
    endDate = int(round(time.time()))
    endDate = endDate - (endDate % dayInUnix)
    return (endDate - dayInUnix * nDays, endDate)

def getAllTags():
    temp_df = pd.read_csv('scraped_data/SP_tags.csv')
    return temp_df['stock'].tolist()

range = getTimestampRange(5 * 365)

df = pd.DataFrame([], columns=['date', 'open', 'high', 'low', 'close', 'volume'], dtype = str)

for stock_tag in getAllTags()[:40]:
    print(stock_tag)

    my_url = "https://query1.finance.yahoo.com/v7/finance/download/{}?period1={}&period2={}&interval=1d&events=history".format(stock_tag, range[0], range[1])
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}

    r = rq.get(url=my_url, headers=headers)

    data = r.text
    data = data.split("\n")
    data = [day.split(",") for day in data][1:]

    temp_df = pd.DataFrame(data, columns =['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume'], dtype = str)
    temp_df = temp_df.drop(['adj_close'], axis=1)
    temp_df['name'] = stock_tag

    df = pd.concat([df, temp_df], ignore_index=True)

def changeDate(date:str):
    dateElems = date.split("-")
    return dateElems[2] + "/" + dateElems[1] + "/" + dateElems[0]
df['date'] = df['date'].apply(changeDate)
    
df.to_csv("scraped_data/output_data.csv", index=False)