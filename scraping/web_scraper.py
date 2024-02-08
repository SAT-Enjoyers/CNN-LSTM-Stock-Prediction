import os
import pandas as pd
import requests as rq
from datetime import datetime as dt
from dateutil.relativedelta import relativedelta

# GLOBAL VARIABLES
DIR = 'scraping/'
OUT_DIR_ALL = os.path.join(DIR, 'SP_data_all_test.csv')
OUT_DIR_SINGLE = os.path.join(DIR, 'SP_data_single.csv')

HEADERS = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}

COLUMNS = ['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']

NUM_STOCKS = 503  # current max = 503

START = 12 * 10  # 10 years of data
END = 0  # actual date or 'today'


def get_relative_timestamp(months: int):
    """
    Gets the timestamp of the date that is `months` months in the past from the current date
    """
    seconds_in_day = 86400

    if isinstance(months, int):
        timestamp = dt.timestamp(dt.now() - relativedelta(months=months))
    else:
        timestamp = dt.strptime(months, '%d/%m/%Y').timestamp()

    # Ensure the timestamp is exactly midnight
    return int(timestamp // seconds_in_day) * seconds_in_day


def get_all_stock_tags():
    """
    Return the tags of all the stocks in S&P 500
    """
    path = os.path.join(DIR, 'SP_tags.csv')
    df = pd.read_csv(path)
    return df['stock'].tolist()


def format_date(date: str):
    """
    Convert format of date
    """
    date_elems = date.split("-")
    return f"{date_elems[2]}/{date_elems[1]}/{date_elems[0]}"


def process_ok_response(r: rq.Response):
    """
    Process the response from the request
    """
    data = r.text

    # Split each day's data and skip the header
    data = data.split("\n")[1:]
    data = [day.split(",") for day in data]

    # Convert raw data to DataFrame
    df = pd.DataFrame(data, columns=COLUMNS, dtype=str)
    df = df.drop(['adj_close'], axis=1)

    return df


def fetch(tags: list[str], start: int, end: int):
    """
    Fetch stock data for the given tags, start, and end timestamps
    """
    # Calculate timestamps
    start_ts = get_relative_timestamp(start)
    end_ts = get_relative_timestamp(end)

    # Initial DataFrame
    df = pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'volume'], dtype=str)

    for i, tag in enumerate(tags):
        print(i, tag)

        # URL setup & GET request
        url = f"https://query1.finance.yahoo.com/v7/finance/download/{tag}?period1={start_ts}&period2={end_ts}&interval=1d&events=history"
        r = rq.get(url=url, headers=HEADERS)

        if r.status_code == 200:
            # Process the response from the request
            temp_df = process_ok_response(r)

            # Add stock tag to each row
            temp_df['name'] = tag

            # Add rows to the main DataFrame
            df = pd.concat([df, temp_df], ignore_index=True)

    df['date'] = df['date'].apply(format_date)

    return df


def fetch_one(start: int, end: int, stock_tag: str):
    """
    Fetch stock data for a single stock tag
    """
    df = fetch([stock_tag], start, end)
    df.to_csv(OUT_DIR_SINGLE, index=False)


def fetch_all(start: int, end: int):
    """
    Fetch stock data for all stock tags
    """
    tags = get_all_stock_tags()[:NUM_STOCKS]
    df = fetch(tags, start, end)
    df.to_csv(OUT_DIR_ALL, index=False)


if __name__ == '__main__':
    fetch_all(START, END)
