from utils.logger import Logger
import utils.config as config

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool
import warnings
import os
import re

logger = Logger().get_logger()
warnings.filterwarnings("ignore")

dividend_df = pd.read_csv(f"{config.path["data_path"]}/dividend.csv")


def __predict_score(stock_id, predict_data, predict_date):
    if not os.path.isfile(f"{config.path["stock_history_path"]}/{stock_id}.csv"):
        return -1
    predict_length = 20
    stock_history = pd.read_csv(f"{config.path["stock_history_path"]}/{stock_id}.csv")
    stock_history["Date"] = pd.to_datetime(stock_history["Date"])
    stock_history.sort_values("Date", inplace=True)

    stock_history = stock_history[stock_history["Date"] >= predict_date]
    stock_history = stock_history.iloc[: predict_length + 1]

    if stock_id == "1101" and len(stock_history) - 1 < predict_length:
        return -2

    predict = predict_data[predict_data["stock_id"] == int(stock_id)]
    if len(predict) == 0:
        return -1

    mae = predict["MAE"].values[0] if "MAE" in predict.columns.to_list() else ""
    volume = stock_history["Volume"].mean() / 1000

    start_date, end_date = stock_history["Date"].iloc[0], stock_history["Date"].iloc[-1]
    dividend_item = dividend_df[dividend_df["stock_code"] == stock_id]
    dividend_item["Date"] = pd.to_datetime(dividend_item["Date"])
    dividend_item = dividend_item[(dividend_item["Date"] >= start_date) & (dividend_item["Date"] <= end_date)]
    dividend = 1 if len(dividend_item["stock_code"]) > 0 else 0

    stock_history["daily_change_pct"] = stock_history["Close"].pct_change()

    if dividend:
        avg1 = stock_history[stock_history["Date"] < dividend_item["Date"].iloc[0]]["Close"].iloc[1:].mean()
        avg2 = stock_history[stock_history["Date"] >= dividend_item["Date"].iloc[0]]["Close"].mean()
        price1 = stock_history["Close"].iloc[0]
        price2 = float(dividend_item["price"].iloc[0])
        length1 = len(stock_history[stock_history["Date"] < dividend_item["Date"].iloc[0]]) - 1
        length2 = len(stock_history[stock_history["Date"] >= dividend_item["Date"].iloc[0]])

        pct1 = (avg1 - price1) / price1
        pct2 = (avg2 - price2) / price2

        increase = ((pct1 * length1) + (pct2 * length2)) / (length1 + length2)

        dividend_date = stock_history[stock_history["Date"] >= dividend_item["Date"].iloc[0]].iloc[0]
        stock_history.loc[stock_history["Date"] == dividend_date["Date"], "daily_change_pct"] = (dividend_date["Close"] - price2) / price2
    else:
        increase = (stock_history["Close"].iloc[1:].mean() - stock_history["Close"].iloc[0]) / stock_history["Close"].iloc[0]

    close_moving = 0
    close_max = stock_history["daily_change_pct"].iloc[1]
    close_min = stock_history["daily_change_pct"].iloc[1]
    max_increase_day = stock_history["daily_change_pct"].iloc[1]
    min_increase_day = stock_history["daily_change_pct"].iloc[1]
    for i in range(1, len(stock_history)):
        close_moving += stock_history["daily_change_pct"].iloc[i]
        close_max = max(close_max, close_moving)
        close_min = min(close_min, close_moving)
        max_increase_day = max(max_increase_day, stock_history["daily_change_pct"].iloc[i])
        min_increase_day = min(min_increase_day, stock_history["daily_change_pct"].iloc[i])

    result = {
        "股票代號": stock_id,
        "預測平均漲幅": predict["Increase"].values[0],
        "實際平均漲幅": increase,
        "平均漲幅誤差": increase - predict["Increase"].values[0],
        "MAE": mae,
        "成交量": volume,
        "預測當日價格": stock_history["Close"].iloc[0],
        "預測期間平均價格": stock_history["Close"].iloc[1:].mean(),
        "預測期間最大漲幅": close_max,
        "預測期間最大跌幅": close_min,
        "預測期間單日最大漲幅": max_increase_day,
        "預測期間單日最大跌幅": min_increase_day,
        "是否有除權息": dividend,
        "總計日期": len(stock_history) - 1,
    }

    return result


def __predict_score_file(version, date, stock_id=None):
    date = pd.to_datetime(date)
    path_date = date.strftime("%y%m%d")
    predict_data_path = f"{config.path["predict_path"]}/predict{path_date}({version}).csv"
    predict_data = pd.read_csv(predict_data_path)
    predict_date = date
    if stock_id is None:
        stock_id_list_df = pd.read_csv(f"{config.path["data_path"]}/stock_info.csv")
        stock_id_list = stock_id_list_df["公司代號"].astype(str)

        data = []
        for stock_id in stock_id_list:
            result = __predict_score(stock_id, predict_data, predict_date)
            if result == -1:
                continue
            elif result == -2:
                return
            data.append(result)

        df = pd.DataFrame(data)
        df.to_excel(f"{config.path["predict_path"]}/predict_score{path_date}({version}).xlsx", index=False)
        output_file = f"{config.path["predict_path"]}/predict_score{path_date}({version}).xlsx"
        with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:
            df.to_excel(writer, sheet_name="Sheet1", index=False)
            workbook = writer.book
            worksheet = writer.sheets["Sheet1"]

            percent_format = workbook.add_format({"num_format": "0.000%"})
            worksheet.set_column(1, 4, None, percent_format)
            worksheet.set_column(8, 11, None, percent_format)

    else:
        result = __predict_score(stock_id)
        print(result)


def show_score():
    file_list = os.listdir(f"{config.path["predict_path"]}/predict_score/")
    increase_range = [[0, 5], [0, 10], [0, 15], [0, 20], [5, 15], [10, 20]]
    count_df = pd.DataFrame(
        index=["0_5", "0_10", "0_15", "0_20", "5_15", "10,_20"],
        columns=["Increase", "Increase_Max", "Increase_Min", "MAE"],
    ).fillna(0)
    file_count = 0
    for file_name in file_list:
        file_count += 1
        df = pd.read_csv(f"./output/predict_score/{file_name}")
        df.drop(columns=["Error"], inplace=True)
        df["Predict_Increase"] = df["Predict_Increase"].str.rstrip("%").astype("float") / 100.0
        for column in count_df.columns:
            df[column] = df[column].str.rstrip("%").astype("float") / 100.0

        df = df.astype(np.float32)
        df.sort_values("Predict_Increase", inplace=True, ascending=False)
        for column in count_df.columns:
            for index, (min_num, max_num) in enumerate(increase_range):
                count_df[column].iloc[index] += df[column].iloc[min_num:max_num].mean()

    count_df = count_df / file_count
    print(count_df)


def predict_score_getting():
    print(f"Predict score getting")
    pattern = re.compile(r"predict(\d{6})\(v(\d+)\).csv")
    folder_path = f"{config.path["predict_path"]}"

    for filename in os.listdir(folder_path):
        match = pattern.match(filename)
        if match:
            date = match.group(1)
            version = match.group(2)
            if not os.path.isfile(f"{config.path["predict_path"]}/predict_score{date}(v{version}).xlsx"):
                __predict_score_file(f"v{version}", f"20{date[0:2]}-{date[2:4]}-{date[4:6]}")
