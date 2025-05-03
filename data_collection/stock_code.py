from utils.logger import Logger
import utils.config as config

import requests
import pandas as pd

logger = Logger().get_logger()


def __trade_history_getting():
    try:
        # Fetch the trade history from the TWSE API
        r = requests.get("https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL")
        if r.status_code == 200:
            trade_history_df = pd.DataFrame(eval(r.text))
            return trade_history_df
    except Exception as e:
        logger.error(f"Errof trade_history_getting : {str(e)}")


def __stock_code_list_getting():
    try:
        trade_history_df = __trade_history_getting()

        r = requests.get("https://openapi.twse.com.tw/v1/opendata/t187ap03_L")
        if r.status_code == 200:
            stock_info_df = pd.DataFrame(eval(r.text))

            # Skip tickers not present in the trade history or with no trade value
            trade_history_df = trade_history_df[trade_history_df["TradeValue"] != ""]
            stock_info_filltered_df = stock_info_df[stock_info_df["公司代號"].isin(trade_history_df["Code"])]

            stock_info_filltered_df.to_csv(f"{config.path["data_path"]}/stock_info.csv", index=False)
    except Exception as e:
        logger.error(f"Errof stock_code_list_getting : {str(e)}")


def __stock_index_code_list_getting():
    try:
        stock_index_info = [
            {"code": "^TWII", "name": "台灣加權指數"},
            {"code": "^AORD", "name": "澳洲普通股指數"},
            {"code": "^HSI", "name": "香港恒生指數"},
            {"code": "^N225", "name": "日經225指數"},
            # {"code": "^DAX", "name": "德國DAX指數},
            {"code": "^FTSE", "name": "富時100指數"},
            {"code": "^NYA", "name": "紐約證券交易所綜合指數"},
            {"code": "^DJI", "name": "道瓊工業指數"},
            {"code": "^GSPC", "name": "S&P 500指數"},
            {"code": "^IXIC", "name": "NASDAQ指數"},
            {"code": "^SOX", "name": "費城半導體指數"},
            {"code": "000001.SS", "name": "上證綜合指數"},
            {"code": "^KS11", "name": "南韓綜合指數"},
            {"code": "^FCHI", "name": "法國CAC指數"},
            {"code": "^STOXX", "name": "泛歐斯托克600指數"},
            {"code": "^RUI", "name": "羅素1000指數"},
            {"code": "^RUT", "name": "羅素2000指數"},
            {"code": "^RUA", "name": "羅素3000指數"},
        ]

        stock_index_info_df = pd.DataFrame(stock_index_info)
        stock_index_info_df.to_csv(f"{config.path["data_path"]}/stock_index_info.csv", index=False)

    except Exception as e:
        logger.error(f"Errof stock_index_code_list_getting : {str(e)}")


def stock_code_getting():
    print("Stock code list getting.")
    __stock_code_list_getting()
    print("Stock index code list getting.")
    __stock_index_code_list_getting()
