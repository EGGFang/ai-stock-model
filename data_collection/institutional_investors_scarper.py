from utils.logger import Logger
import utils.config as config

import datetime
import pandas as pd
import numpy as np
import os
import re
import requests
from tqdm import tqdm
import time
import random

logger = Logger().get_logger()


def __is_subsequence_in_string(s1, s2):
    pattern = ".*".join(re.escape(char) for char in s1)

    return re.search(pattern, s2) is not None


def __institutional_getting(date):
    check = 0
    while True:
        try:
            a = {
                "date": date.strftime("%Y%m%d"),
                "selectType": "ALL",
                "response": "json",
                "_": "0",
            }

            r = requests.get(f"https://www.twse.com.tw/rwd/zh/fund/T86", params=a, headers={"Connection": "close"}, timeout=60)
            if r.json()["stat"] == "OK":
                data = r.json()["data"]
                fields = r.json()["fields"]
                fields = [s.replace("</br>", "") for s in fields]
                df = pd.DataFrame(data, columns=fields).replace(",", "", regex=True).replace("--", np.nan).replace("</br>", "", regex=True)
            elif r.json()["stat"] == "很抱歉，沒有符合條件的資料!":
                return pd.DataFrame()
            else:
                raise Exception(f"Message:{r.json()["stat"]}")

            columns = [
                "外資買進股數",
                "外資賣出股數",
                "外資買賣超股數",
                "投信買進股數",
                "投信賣出股數",
                "投信買賣超股數",
                "自營商買賣超股數",
                "自營商買進股數",
                "自營商賣出股數",
                "三大法人買賣超股數",
            ]
            for column in columns:
                df_columns_list = df.columns.to_list()
                if not (column in df_columns_list):
                    columns_list = [df_column for df_column in df_columns_list if __is_subsequence_in_string(column, df_column)]
                    df[column] = df[columns_list].astype(float).sum(axis=1)
                    df.drop(columns=columns_list, inplace=True)

            columns.append("證券代號")
            df = df[columns]
            df.rename(columns={"證券代號": "stock_code"}, inplace=True)
            df.insert(0, "Date", date)
            df["Date"] = pd.to_datetime(df["Date"])
            check = 0

            return df

        except Exception as e:
            if check < 5:
                check += 1
                time.sleep(10)
            else:
                logger.error(f"Error institutional_getting: {str(e)}")
                return


def institutional_investors_getting():
    print("Getting institutional investors")
    start_date = datetime.date(year=2012, month=5, day=2)
    old_df = pd.DataFrame()
    stock_id_list_df = pd.read_csv(f"{config.path["data_path"]}/stock_info.csv")
    stock_id_list = stock_id_list_df["公司代號"].astype(str)
    if os.path.isfile(f"{config.path["data_path"]}/institutional.csv"):
        old_df = pd.read_csv(f"{config.path["data_path"]}/institutional.csv")
        old_df["Date"] = pd.to_datetime(old_df["Date"])
        old_df.sort_values("Date", inplace=True)
        start_date = old_df["Date"].iloc[-1]
        start_date = datetime.date(year=start_date.year, month=start_date.month, day=start_date.day)

    date_list = [(start_date + datetime.timedelta(days=x)) for x in range((datetime.date.today() - start_date).days)]
    tqdm_bar = tqdm(total=len(date_list))
    for i in range(len(date_list)):
        new_df = __institutional_getting(date_list[i])
        tqdm_bar.update(1)
        if len(new_df) == 0:
            continue

        new_df = new_df[new_df["stock_code"].isin(stock_id_list)]
        old_df = pd.concat([old_df, new_df])
        time.sleep(random.randint(5, 10))

        if i % 10 == 0:
            old_df.drop_duplicates(subset=["Date", "stock_code"], inplace=True)
            old_df["Date"] = pd.to_datetime(old_df["Date"])
            old_df.sort_values("Date", inplace=True)
            old_df.to_csv(f"{config.path["data_path"]}/institutional.csv", index=False)

    old_df.drop_duplicates(subset=["Date", "stock_code"], inplace=True)
    old_df["Date"] = pd.to_datetime(old_df["Date"])
    old_df.sort_values("Date", inplace=True)
    old_df.to_csv(f"{config.path["data_path"]}/institutional.csv", index=False)
