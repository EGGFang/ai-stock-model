from utils.logger import Logger
import utils.config as config

from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import requests
import pandas as pd
import datetime
import re
import time
import os
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

logger = Logger().get_logger()


def __convert_date_format(date_str):
    # Convert date string from "YYYY年MM月DD日" to "YYYY/MM/DD" format
    match = re.match(r"(\d+)年(\d+)月(\d+)日", date_str)
    year = match.group(1)
    month = match.group(2)
    day = match.group(3)
    return f"{int(year)+1911}/{month}/{day}"


def __dividend_history_getting(start_date):
    try:
        r = requests.get(
            f"https://www.twse.com.tw/rwd/zh/exRight/TWT49U?startDate={start_date.strftime('%Y%m%d')}&endDate={datetime.datetime.today().strftime('%Y%m%d')}&response=json&_=000"
        )
        if r.status_code == 200:
            dividend_history_df = pd.DataFrame(r.json()["data"], columns=r.json()["fields"])

            dividend_history_df["資料日期"] = dividend_history_df["資料日期"].apply(__convert_date_format)
            dividend_history_df = dividend_history_df[["資料日期", "股票代號", "權/息", "除權息參考價"]]
            dividend_history_df.rename(columns={"資料日期": "Date", "股票代號": "stock_code", "權/息": "type", "除權息參考價": "price"}, inplace=True)

            return dividend_history_df
    except Exception as e:
        logger.error(f"Error dividend_history_getting: {str(e)}")


def __dividend_preview_getting():
    try:
        r = requests.get(f"https://www.twse.com.tw/rwd/zh/exRight/TWT48U?response=json&_=000")
        if r.status_code == 200:
            dividend_preview_df = pd.DataFrame(r.json()["data"], columns=r.json()["fields"])

            dividend_preview_df["除權除息日期"] = dividend_preview_df["除權除息日期"].apply(__convert_date_format)
            dividend_preview_df = dividend_preview_df[["除權除息日期", "股票代號", "除權息"]]
            dividend_preview_df.rename(columns={"除權除息日期": "Date", "股票代號": "stock_code", "除權息": "type"}, inplace=True)

            return dividend_preview_df
    except Exception as e:
        logger.error(f"Error dividend_preview_getting: {str(e)}")


def dividend_getting():
    try:
        print("Stock dividend getting.")
        start_date = datetime.date(2010, 1, 1)

        diviend_history_df = __dividend_history_getting(start_date)
        dividend_preview_df = __dividend_preview_getting()

        dividend_df = pd.concat([diviend_history_df, dividend_preview_df])
        dividend_df.drop_duplicates(subset=["Date", "stock_code"], inplace=True)
        dividend_df["Date"] = pd.to_datetime(dividend_df["Date"])
        dividend_df.sort_values(by="Date", inplace=True)
        dividend_df["price"] = dividend_df["price"].replace(",", "", regex=True)
        dividend_df.to_csv(f"{config.path["data_path"]}/dividend.csv", index=False)

    except Exception as e:
        logger.error(f"Error dividend_getting: {str(e)}")
