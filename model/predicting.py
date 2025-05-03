from .model_scoring import ma_predict
from .StackingAveragedModels import StackingAveragedModels
from data_processing.data_build import __read_data, __add_new_columns, __data_grouping
from utils.logger import logging
from utils import config

import pickle
from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd
import numpy as np
import warnings
import os

warnings.filterwarnings("ignore")

version = config.predicting["version"]
statistics_df = pd.read_csv(f"{config.path["changelog_path"]}/{version}/statistics.csv")


def __predict(stock_id):
    try:
        score_df = pd.read_csv(f"{config.path["changelog_path"]}/{version}/score.csv")
        # check model exist?
        if not os.path.isfile(f"{config.path["model_path"]}/{version}/{stock_id}_model.pkl"):
            return

        stock_history = pd.read_csv(f"{config.path["news_scored_path"]}/{stock_id}.csv")
        statistics = statistics_df[statistics_df["stock_id"].astype(str) == stock_id]
        if len(stock_history) < 30 or len(statistics) == 0:
            return

        stock_history["Date"] = pd.to_datetime(stock_history["Date"])
        stock_history.sort_values("Date", inplace=True)
        stock_history.set_index(stock_history["Date"], inplace=True)

        model = pickle.load(open(f"{config.path["model_path"]}/{version}/{stock_id}_model.pkl", "rb"))
        sc = pickle.load(open(f"{config.path["model_path"]}/{version}/{stock_id}_sc.pkl", "rb"))
        data = __read_data(stock_id)
        if data is None:
            return None
        data = __add_new_columns(data)
        data = __data_grouping(stock_id, data, "Test")
        data = np.array(data)
        data = sc.transform(data)
        model_predict = model.predict(data)
        model_predict = ma_predict(model_predict)

        stock_history = stock_history.iloc[-30:]
        # draw(stock_id, stock_history)
        mae = score_df[score_df[score_df.columns[0]] == int(stock_id)]["MAE"].values[0]

        return {
            "stock_id": stock_id,
            "Increase": model_predict[-1],
            "Training_Normalize_score": (model_predict[-1] - statistics["training_mean"].iloc[0]) / statistics["training_std"].iloc[0],
            "Testing_Normalize_score": (model_predict[-1] - statistics["testing_mean"].iloc[0]) / statistics["testing_std"].iloc[0],
            "MAE": mae,
            "Volume": stock_history["Volume"].mean() / 1000,
        }

    except Exception as e:
        print(f"{e}, {stock_id}")


def predict_getting(p=8):
    __predict("1256")
    print(f"Predicting {version}")
    stock_id_list_df = pd.read_csv(f"{config.path["data_path"]}/stock_info.csv")
    stock_id_list = stock_id_list_df["公司代號"].astype(str)
    result_df = pd.DataFrame(columns=["stock_id", "Increase", "MAE"])
    p = Pool(processes=p)
    tqdm_bar = tqdm(total=len(stock_id_list))
    for result in p.imap(__predict, stock_id_list):
        result_df = result_df._append(result, ignore_index=True)
        tqdm_bar.update(1)

    data = pd.read_csv(f"{config.path["news_scored_path"]}/1101.csv")
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date")
    now_time = data["Date"].iloc[-1]
    date = now_time.strftime("%y%m%d")

    result_df.to_csv(f"{config.path["predict_path"]}/predict{date}({version}).csv", index=False)
