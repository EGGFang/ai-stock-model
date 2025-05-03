from utils.logger import Logger
import utils.config as config

from multiprocessing import Pool
import dl_translate as dlt
import pandas as pd
from tqdm import tqdm
import os

logger = Logger().get_logger()

mt = dlt.TranslationModel("mbart50")


def __translate_to_english_dl(text, batch_size=1):
    try:
        # Translate text to English(dl_translate)
        result = mt.translate(text, source=dlt.lang.CHINESE, target=dlt.lang.ENGLISH, batch_size=batch_size)
        return result

    except Exception as e:
        logger.error(f"Error translate_to_english_dl: {str(e)}")


def file_translate(file_name):
    try:
        # Read the original CSV file
        df = pd.read_csv(f"{config.path["news_text_path"]}/{file_name}")

        if len(df) == 0:
            return

        if not ("title_eng" in df.columns.to_list()):
            df["title_eng"] = ""
            df["summary_eng"] = ""

        # Translate title column for news
        for i in range(len(df)):
            if pd.isna(df["title_eng"].iloc[i]):
                df["title_eng"].iloc[i] = __translate_to_english_dl(df["title"].iloc[i])

        # Translate summary columns for Yahoo news
        if "yahoo" in file_name:
            for i in range(len(df)):
                if pd.isna(df["summary_eng"].iloc[i]):
                    df["summary_eng"].iloc[i] = __translate_to_english_dl(df["summary"].iloc[i])

        df.dropna(inplace=True)
        # Save the translated DataFrame to a CSV file
        df.to_csv(f"{config.path["news_text_path"]}/{file_name}", index=False)

    except Exception as e:
        logger.error(f"Error file_translate: {str(e)}")


def translate_news():
    try:
        print("Stock news translating.")
        # file_translate("yuanta2712.csv")
        file_list = os.listdir(config.path["news_text_path"])
        tqdm_bar = tqdm(total=len(file_list), smoothing=0)
        for file_name in file_list:
            file_translate(file_name)
            tqdm_bar.update(1)

    except Exception as e:
        logger.error(f"Error translate_news: {str(e)}")
