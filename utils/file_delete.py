from utils import config
from utils.logger import Logger

import os
import re
import warnings

logger = Logger().get_logger()
warnings.filterwarnings("ignore")


def model_delete(stock_id=None):
    folder_path = config.path["model_path"]
    pattern = re.compile(r"^\d+_model\.pkl$")

    for filename in os.listdir(folder_path):
        if pattern.match(filename):
            if stock_id is None or filename == f"{stock_id}_model.pkl":
                file_path = os.path.join(folder_path, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)


def params_delete():
    folder_path = config.path["model_path"]
    file_path = os.path.join(folder_path, "best_params.pkl")
    if os.path.isfile(file_path):
        os.remove(file_path)


def training_data_delete(stock_id=None):
    folder_path = config.path["training_data_path"]

    for filename in os.listdir(folder_path):
        if stock_id is None or filename == f"{stock_id}.npy":
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
