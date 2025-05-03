from utils.logger import Logger
import utils.config as config
from .StackingAveragedModels import StackingAveragedModels
from .paramter_optimization import get_best_params

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.utils import shuffle
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
import lightgbm as lgb
import xgboost
import catboost

from multiprocessing import Pool
import pickle
from tqdm import tqdm
import pandas as pd
import numpy as np
import random
import warnings
import os


logger = Logger().get_logger()
random.seed(10)
warnings.filterwarnings("ignore")


def training(stock_id, X_train, Y_train):
    try:
        best_params = pickle.load(open(f"{config.path["model_path"]}/best_params.pkl", "rb"))
        params_dict = {
            "LightGBM": [
                "num_leaves",
                "n_estimators",
                "max_bin",
                "bagging_freq",
                "min_data_in_leaf",
            ],
            "Lasso": ["max_iter"],
            "RandomForest": ["n_estimators", "max_depth"],
            "ElasticNet": ["max_iter"],
            "Ridge": [],
        }

        model_list = []
        for key in list(params_dict.keys()):
            if key in best_params:
                best_params[key] = best_params[key]["params"]
                for params in params_dict[key]:
                    best_params[key][params] = int(best_params[key][params])

        if config.training["paramter_opt_model"]["ElasticNet"]:
            ENet = make_pipeline(RobustScaler(), ElasticNet(**best_params["ElasticNet"], random_state=42))
            model_list.append(ENet)

        if config.training["paramter_opt_model"]["Lasso"]:
            lasso = make_pipeline(RobustScaler(), Lasso(**best_params["Lasso"], random_state=42))
            model_list.append(lasso)

        if config.training["paramter_opt_model"]["RandomForest"]:
            best_params["RandomForest"]["max_features"] = "sqrt"
            rf = RandomForestRegressor(
                **best_params["RandomForest"],
                random_state=42,
            )
            model_list.append(rf)

        if config.training["paramter_opt_model"]["LightGBM"]:
            best_params["LightGBM"]["verbosity"] = -1
            best_params["LightGBM"]["metric"] = "l1"
            # best_params["LightGBM"]["early_stopping_round"] = 50
            # best_params["LightGBM"]["device"] = "gpu"
            model_lgb = lgb.LGBMRegressor(
                **best_params["LightGBM"],
                random_state=42,
            )
            model_list.append(model_lgb)

        if config.training["paramter_opt_model"]["Ridge"]:
            best_params["Ridge"]["solver"] = "lbfgs"
            best_params["Ridge"]["positive"] = True
            model_rr = Ridge(
                **best_params["Ridge"],
                random_state=42,
            )
            model_list.append(model_rr)

        if os.path.isfile(f"{config.path["model_path"]}/{stock_id}_model.pkl"):
            stacked_averaged_models = pickle.load(open(f"{config.path["model_path"]}/{stock_id}_model.pkl", "rb"))
        else:
            meta_model = xgboost.XGBRegressor(random_seed=42)
            stacked_averaged_models = StackingAveragedModels(base_models=tuple(model_list), meta_model=meta_model)
            stacked_averaged_models.fit(X_train, Y_train[:, 0])

        pickle.dump(stacked_averaged_models, open(f"{config.path["model_path"]}/{stock_id}_model.pkl", "wb"))
        return

    except Exception as e:
        logger.error(f"Error training: {str(e)}")


def model_processing(stock_id):
    try:
        if os.path.isfile(f"{config.path["model_path"]}/{stock_id}_model.pkl"):
            return

        if os.path.isfile(f"{config.path["training_data_path"]}/{stock_id}.npy"):
            data = np.load(f"{config.path["training_data_path"]}/{stock_id}.npy")
        else:
            return

        test_data_length = config.training["test_data_length"]

        X_train = data[:-test_data_length, :-2]
        X_test = data[-test_data_length:, :-2]
        Y_train = data[:-test_data_length, -2:]
        Y_test = data[-test_data_length:, -2:]
        X_train, Y_train = shuffle(X_train, Y_train, random_state=42)

        training(stock_id, X_train, Y_train)

        return

    except Exception as e:
        logger.error(f"Error model_processing: {str(e)}")


def model_training(process, stock_id_list):
    try:
        print("Model training.")
        tqdm_bar = tqdm(total=len(stock_id_list), smoothing=0)
        # model_processing("1101")
        if process == 1:
            for stock_id in stock_id_list:
                result = model_processing(stock_id)
                tqdm_bar.update(1)

        elif process > 1:
            p = Pool(process)
            for result in p.imap(model_processing, stock_id_list):
                tqdm_bar.update(1)

    except Exception as e:
        logger.error(f"Error model_training: {str(e)}")


def best_params_getting(stock_id_list):
    try:
        if os.path.isfile(f"{config.path["model_path"]}/best_params.pkl"):
            return

        print("Best params getting.")
        test_data_length = config.training["test_data_length"]
        X_train = X_test = Y_train = Y_test = None
        tqdm_bar = tqdm(total=len(stock_id_list))
        for stock_id in stock_id_list:
            tqdm_bar.update(1)

            if os.path.isfile(f"{config.path["training_data_path"]}/{stock_id}.npy"):
                data = np.load(f"{config.path["training_data_path"]}/{stock_id}.npy")
            else:
                continue

            if X_train is None:
                X_train = data[:-test_data_length, :-2]
                X_test = data[-test_data_length:, :-2]
                Y_train = data[:-test_data_length, -2:]
                Y_test = data[-test_data_length:, -2:]
            else:
                X_train = np.concatenate((X_train, data[:-test_data_length, :-2]), axis=0)
                X_test = np.concatenate((X_test, data[-test_data_length:, :-2]), axis=0)
                Y_train = np.concatenate((Y_train, data[:-test_data_length, -2:]), axis=0)
                Y_test = np.concatenate((Y_test, data[-test_data_length:, -2:]), axis=0)

        best_params = get_best_params(X_train, Y_train[:, 0], X_test, Y_test[:, 0])
        pickle.dump(best_params, open(f"{config.path["model_path"]}/best_params.pkl", "wb"))

    except Exception as e:
        logger.error(f"Error best_params_getting: {str(e)}")


def get_model(process=1, stock_id_list=None, best_params=True):
    try:
        if stock_id_list is None:
            stock_id_list_df = pd.read_csv(f"{config.path["data_path"]}/stock_info.csv")
            stock_id_list = stock_id_list_df["公司代號"].astype(str)

        if best_params:
            best_params_getting(random.sample(list(stock_id_list), min(100, len(stock_id_list))))

        model_training(process, stock_id_list)

    except Exception as e:
        logger.error(f"Error get_model: {str(e)}")


if __name__ == "__main__":
    # main("2722")
    model_training()
    # get_model()
