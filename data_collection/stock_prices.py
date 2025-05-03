from utils.logger import Logger
import utils.config as config
from utils.driver_manager import DriverManager

from bs4 import BeautifulSoup
from multiprocessing import Pool
import FinMind
import yfinance as yf
import pandas as pd
import datetime
import requests
from tqdm import tqdm
import os
import time
import random
import json

logger = Logger().get_logger()
stock_info_df = pd.read_csv(f"{config.path["data_path"]}/stock_info.csv")


def __yahoo_data_getting(code_list, end_string=""):
    try:
        start_date = "2010-01-01"
        tqdm_bar = tqdm(total=len(code_list))
        # Retrieve stock data from Yahoo Finance for the given code list
        for ticker in code_list:
            # Download data from Yahoo Finance
            data = yf.download(
                tickers=f"{ticker}{end_string}",
                start=start_date,
                end=datetime.date.today(),
                progress=False,
            )
            data.reset_index(inplace=True)
            data.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]
            data.to_csv(f"{config.path["stock_history_path"]}/{ticker}.csv", index=False)
            tqdm_bar.update(1)

        tqdm_bar.close()

    except Exception as e:
        logger.error(f"Error yahoo_data_getting: {str(e)}")


def __tw_stock_future_getting():
    try:
        dl = FinMind.data.DataLoader()
        future_data = dl.taiwan_futures_daily(futures_id="TX", start_date="2010-01-01")
        future_data = future_data[future_data["trading_session"] == "position"]
        future_data.drop(columns=["futures_id", "spread", "settlement_price", "open_interest", "trading_session"], inplace=True)
        future_data = future_data[future_data["spread_per"] != 0]

        future_data.to_csv(f"{config.path["data_path"]}/future.csv", index=False)

    except Exception as e:
        logger.error(f"Error tw_stock_future_getting: {str(e)}")


def __gov_convert_date(date):
    parts = date.split("/")
    return f"{int(parts[0])+1911}/{parts[1]}/{parts[2]}"


def __gov_stock_history_fetch(dt, stock_id, driver=None):
    check = 0
    while True:
        try:
            if driver is None:
                request = requests.get(
                    f"https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY?date={dt.strftime("%Y%m%d")}&stockNo={stock_id}&response=json", timeout=20
                )
                json_data = request.json()
            else:
                driver.get(f"https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY?date={dt.strftime("%Y%m%d")}&stockNo={stock_id}&response=json")
                time.sleep(2)
                html = driver.page_source
                soup = BeautifulSoup(html, "html.parser")
                s = str(soup.find("pre"))
                json_str = s.replace("<pre>", "").replace("</pre>", "")
                if json_str is None or len(json_str) == 0:
                    raise Exception(f"Message:{json_str}")
                json_data = json.loads(json_str)

            if json_data["stat"] == "OK":
                data = json_data["data"]
                fields = json_data["fields"]
                df = pd.DataFrame(data, columns=fields)
                df = df[["日期", "成交股數", "開盤價", "最高價", "最低價", "收盤價", "成交筆數"]]
                df.rename(
                    columns={
                        "日期": "Date",
                        "成交股數": "Volume",
                        "開盤價": "Open",
                        "最高價": "High",
                        "最低價": "Low",
                        "收盤價": "Close",
                        "成交筆數": "trade_count",
                    },
                    inplace=True,
                )
                df["Date"] = df["Date"].apply(__gov_convert_date)
                for column in ["Volume", "Open", "High", "Low", "Close", "trade_count"]:
                    df[column] = df[column].replace(",", "", regex=True)

                check = 0
                return df

            elif json_data["stat"] == "查詢日期大於今日，請重新查詢!" or json_data["stat"] == "很抱歉，沒有符合條件的資料!":
                if check > 0:
                    return pd.DataFrame()
                else:
                    check += 1
                    time.sleep(random.randint(4, 7))
                    continue
            else:
                raise Exception(f"Message:{json_data["stat"]}")

        except Exception as e:
            if check < 5:
                check += 1
                time.sleep(random.randint(5, 10))
            else:
                logger.error(f"Error gov_stock_history_fetch: {str(e)}, {dt.strftime("%Y%m%d")}, {stock_id}, {json_str}")
                raise Exception(f"Message:{json_data["stat"]}")


def __gov_stock_history(stock_id_list):
    try:
        tqdm_bar = tqdm(total=len(stock_id_list))
        for stock_id in stock_id_list:
            old_df = pd.DataFrame()
            year, month = 2010, 1
            if os.path.isfile(f"{config.path["stock_history_path"]}/{stock_id}.csv"):
                old_df = pd.read_csv(f"{config.path["stock_history_path"]}/{stock_id}.csv")
                old_df["Date"] = pd.to_datetime(old_df["Date"])
                old_df.sort_values("Date", inplace=True)
                year, month = old_df["Date"].iloc[-1].year, old_df["Date"].iloc[-1].month

            now_datetime = datetime.datetime.now()
            while True:
                dt = datetime.datetime(year, month, 1)
                if dt > now_datetime:
                    break

                df = __gov_stock_history_fetch(dt, stock_id)

                old_df = pd.concat([old_df, df])

                month += 1
                if month > 12:
                    month = 1
                    year += 1

                    if len(old_df) > 0:
                        old_df["Date"] = pd.to_datetime(old_df["Date"])
                        old_df.sort_values("Date", inplace=True)
                        old_df.drop_duplicates(subset=["Date"], inplace=True)
                        old_df.to_csv(f"{config.path["stock_history_path"]}/{stock_id}.csv", index=False)

                time.sleep(random.randint(2, 3))

            old_df["Date"] = pd.to_datetime(old_df["Date"])
            old_df.sort_values("Date", inplace=True)
            old_df.drop_duplicates(subset=["Date"], inplace=True)
            old_df.to_csv(f"{config.path["stock_history_path"]}/{stock_id}.csv", index=False)

            tqdm_bar.update(1)

    except Exception as e:
        logger.error(f"Error gov_stock_history: {str(e)}")


def __gov_get_driver(stock_id):
    check = 0
    while True:
        try:
            driver = DriverManager().get_driver(vpn=True, headless=True)
            if driver is None:
                continue
            driver.get(f"https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY?date=20240101&stockNo={stock_id}&response=json")
            html = driver.page_source
            soup = BeautifulSoup(html, "html.parser")
            s = str(soup.find("pre"))
            json_str = s.replace("<pre>", "").replace("</pre>", "")
            json_data = json.loads(json_str)
            if json_data["stat"]:
                time.sleep(random.randint(2, 3))
                break
        except Exception as e:
            time.sleep(random.randint(5, 15))
            try:
                driver.quit()
            except:
                pass
            check += 1
            if check > 4:
                logger.error(f"Error gov_get_driver({stock_id}): {str(e)}")

    return driver


def __gov_stock_history_driver(stock_id):
    try:
        old_df = pd.DataFrame()
        year, month = 2010, 1
        now_datetime = datetime.datetime.now()

        if os.path.isfile(f"{config.path["stock_history_path"]}/{stock_id}.csv"):
            old_df = pd.read_csv(f"{config.path["stock_history_path"]}/{stock_id}.csv")
            old_df["Date"] = pd.to_datetime(old_df["Date"])
            old_df.sort_values("Date", inplace=True)
            year, month = old_df["Date"].iloc[-1].year, old_df["Date"].iloc[-1].month
            if old_df["Date"].iloc[-1].date() == now_datetime.date():
                return
        else:
            if len(stock_info_df[stock_info_df["公司代號"].astype(str) == stock_id]) > 0:
                start_date = str(stock_info_df[stock_info_df["公司代號"].astype(str) == stock_id]["上市日期"].iloc[0])
                if len(start_date) == 8:
                    start_year, start_month = int(start_date[0:4]), int(start_date[4:6])
                    if year < start_year or (year == start_year and month < start_month):
                        year, month = start_year, start_month

        check = 0
        while True:
            try:
                driver = __gov_get_driver(stock_id)
                while True:
                    dt = datetime.datetime(year, month, 1)
                    if dt > now_datetime:
                        break

                    df = __gov_stock_history_fetch(dt, stock_id, driver)
                    old_df = pd.concat([old_df, df])
                    time.sleep(random.randint(4, 7))

                    month += 1
                    if month > 12:
                        month = 1
                        year += 1

                        if len(old_df) > 0:
                            old_df["Date"] = pd.to_datetime(old_df["Date"])
                            old_df.sort_values("Date", inplace=True)
                            old_df.drop_duplicates(subset=["Date"], inplace=True)
                            old_df.to_csv(f"{config.path["stock_history_path"]}/{stock_id}.csv", index=False)

                old_df["Date"] = pd.to_datetime(old_df["Date"])
                old_df.sort_values("Date", inplace=True)
                old_df.drop_duplicates(subset=["Date"], inplace=True)
                old_df["Close"] = pd.to_numeric(old_df["Close"], errors="coerce")
                old_df.dropna(subset=["Close"], inplace=True)
                old_df.to_csv(f"{config.path["stock_history_path"]}/{stock_id}.csv", index=False)

                try:
                    driver.quit()
                except:
                    pass
                return

            except Exception as e:
                try:
                    driver.quit()
                except:
                    pass
                check += 1
                if check > 4:
                    logger.error(f"Error gov_stock_history_driver({stock_id}, {year}, {month}): {str(e)}")

    except Exception as e:
        try:
            driver.quit()
        except:
            pass
        logger.error(f"Error gov_stock_history_driver({stock_id}, {year}, {month}): {str(e)}")


def stock_data_getting():
    try:
        print("Stock data getting.")
        # Retrieve stock codes from the database
        stock_index_code_df = pd.read_csv(f"{config.path["data_path"]}/stock_index_info.csv")
        stock_code_list = stock_info_df["公司代號"].to_list()
        stock_index_code_list = stock_index_code_df["code"].to_list()

        # Fetch data for stocks, indices, and ETFs
        __gov_stock_history_driver("1101")

        p = Pool(processes=6)
        tqdm_bar = tqdm(total=len(stock_code_list), smoothing=0)
        for result in p.imap(__gov_stock_history_driver, stock_code_list):
            tqdm_bar.update(1)
        tqdm_bar.close()

        # __gov_stock_history(stock_code_list)

        __yahoo_data_getting(code_list=stock_index_code_list)
        __tw_stock_future_getting()

    except Exception as e:
        logger.error(f"Error stock_data_getting: {str(e)}")
