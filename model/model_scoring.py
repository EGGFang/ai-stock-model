from utils.logger import Logger
import utils.config as config
from .StackingAveragedModels import StackingAveragedModels

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from multiprocessing import Pool
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from tqdm import tqdm
import pandas as pd
import numpy as np
import warnings
import os

logger = Logger().get_logger()
warnings.filterwarnings("ignore")
matplotlib.use("Agg")
sns.set_style("darkgrid")


def scores(stock_id, predict_Regressor, predict_Classifier, Y_Regressor, Y_Classifier):
    try:
        acc = accuracy_score(Y_Classifier, predict_Classifier)
        recall = recall_score(Y_Classifier, predict_Classifier)
        precision = precision_score(Y_Classifier, predict_Classifier)
        f1 = f1_score(Y_Classifier, predict_Classifier)
        roc = -1

        mse = mean_squared_error(Y_Regressor, predict_Regressor)
        mae = mean_absolute_error(Y_Regressor, predict_Regressor)
        r2 = r2_score(Y_Regressor, predict_Regressor)

        result = {
            "stock_id": stock_id,
            "Acc": acc,
            "Recall": recall,
            "Precision": precision,
            "F1": f1,
            "AUC": roc,
            "MSE": mse,
            "MAE": mae,
            "R2": r2,
        }
        return result

    except Exception as e:
        logger.error(f"Error scores: {str(e)}")


def ma_predict(predict_Regressor):
    try:
        predict_window_length = config.training["predict_window_length"]
        ma_predict_Regressor = np.empty_like(predict_Regressor, dtype=float)
        for i in range(len(predict_Regressor)):
            if i < (predict_window_length - 1):
                ma_predict_Regressor[i] = np.mean(predict_Regressor[: i + 1])
            else:
                ma_predict_Regressor[i] = np.mean(predict_Regressor[i - (predict_window_length - 1) : i + 1])

        return ma_predict_Regressor

    except Exception as e:
        logger.error(f"Error ma_predict: {str(e)}")


def increase(stock_predict, statistics_list):
    def increase_count(sort_type, stock_predict):
        increase_top5 = []
        increase_df[sort_type] = 0
        for i in range(test_data_length):
            sorted_indices = np.argsort(stock_predict[:, 0, i])
            increase_df[sort_type].iloc[0] += np.mean(stock_predict[sorted_indices[-5:], 1, i])  # Increase top 5
            increase_df[sort_type].iloc[1] += np.mean(stock_predict[sorted_indices[-10:], 1, i])  # Increase top 10
            increase_df[sort_type].iloc[2] += np.mean(stock_predict[sorted_indices[-15:], 1, i])  # Increase top 15
            increase_df[sort_type].iloc[3] += np.mean(stock_predict[sorted_indices[-20:], 1, i])  # Increase top 20
            increase_df[sort_type].iloc[4] += np.mean(stock_predict[sorted_indices[-15:-5], 1, i])  # Increase top 15 to 5
            increase_df[sort_type].iloc[5] += np.mean(stock_predict[sorted_indices[-20:-10], 1, i])  # Increase top 20 to 10

            increase_top5.append(np.mean(stock_predict[sorted_indices[-5:], 1, i]))

        sns.displot(increase_top5)
        plt.savefig(f"{config.path["output_path"]}/predict_{sort_type}_top5.png")

    test_data_length = config.training["test_data_length"]
    increase_df = pd.DataFrame({"type": ["Increase_0_5", "Increase_0_10", "Increase_0_15", "Increase_0_20", "Increase_5_15", "Increase_10_20"]})
    sort_type_list = ["Normal", "Normalization_Training", "Normalization_Testing"]

    train_nor_stock_predict = stock_predict.copy()
    test_nor_stock_predict = stock_predict.copy()
    for i in range(len(statistics_list)):
        train_nor_stock_predict[i, 0, :] -= statistics_list[i]["training_mean"]
        train_nor_stock_predict[i, 0, :] /= statistics_list[i]["training_std"]
    for i in range(len(statistics_list)):
        test_nor_stock_predict[i, 0, :] -= statistics_list[i]["testing_mean"]
        test_nor_stock_predict[i, 0, :] /= statistics_list[i]["testing_std"]

    increase_count("Normal", stock_predict)
    increase_count("Normalization_Training", train_nor_stock_predict)
    increase_count("Normalization_Testing", test_nor_stock_predict)

    for sort_type in sort_type_list:
        increase_df[sort_type] /= test_data_length

    increase_df.to_csv(f"{config.path["output_path"]}/increase.csv", index=False)


def model_score(stock_id):
    if not os.path.isfile(f"{config.path["model_path"]}/{stock_id}_model.pkl"):
        return None, None, None

    if os.path.isfile(f"{config.path["training_data_path"]}/{stock_id}.npy"):
        data = np.load(f"{config.path["training_data_path"]}/{stock_id}.npy")
    else:
        return None, None, None

    test_data_length = config.training["test_data_length"]

    X_train = data[:-test_data_length, :-2]
    X_test = data[-test_data_length:, :-2]
    Y_train = data[:-test_data_length, -2:]
    Y_test = data[-test_data_length:, -2:]

    stacked_averaged_models = pickle.load(open(f"{config.path["model_path"]}/{stock_id}_model.pkl", "rb"))

    training_data_predictions = stacked_averaged_models.predict(X_train)
    testing_data_predictions = stacked_averaged_models.predict(X_test)
    predict_Regressor = testing_data_predictions
    ma_predict_Regressor = ma_predict(predict_Regressor)

    predict_Classifier = np.where(ma_predict_Regressor > 0, 1, 0)

    predict_data = np.vstack((predict_Regressor.reshape(1, -1), Y_test[:, -2].reshape(1, -1))).reshape(1, 2, test_data_length)
    score = scores(stock_id, ma_predict_Regressor, predict_Classifier, Y_test[:, 0], Y_test[:, 1])

    statistics = {
        "stock_id": stock_id,
        "training_mean": np.mean(training_data_predictions),
        "testing_mean": np.mean(testing_data_predictions),
        "training_std": np.std(training_data_predictions),
        "testing_std": np.std(testing_data_predictions),
    }

    return predict_data, score, statistics


def model_score_getting(stock_id_list=None):
    if stock_id_list is None:
        stock_id_list_df = pd.read_csv(f"{config.path["data_path"]}/stock_info.csv")
        stock_id_list = stock_id_list_df["公司代號"].astype(str)
    test_data_length = config.training["test_data_length"]

    p = Pool(4)
    tqdm_bar = tqdm(total=len(stock_id_list), smoothing=0)
    stock_predict = np.empty((0, 2, test_data_length))
    score_list = []
    statistics_list = []
    for predict_data, score, statistics in p.imap(model_score, stock_id_list):
        if not (predict_data is None):
            stock_predict = np.vstack((stock_predict, predict_data))
            score_list.append(score)
            statistics_list.append(statistics)
        tqdm_bar.update(1)

    increase(stock_predict, statistics_list)
    score_df = pd.DataFrame(score_list)
    score_df.to_csv(f"{config.path["output_path"]}/score.csv", index=False)
    statistics_df = pd.DataFrame(statistics_list)
    statistics_df.to_csv(f"{config.path["output_path"]}/statistics.csv", index=False)
