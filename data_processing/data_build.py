from utils.logger import Logger
import utils.config as config

from sklearn.preprocessing import StandardScaler
from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle
import os
import warnings

logger = Logger().get_logger()
warnings.filterwarnings("ignore")


def __calc_ema(data, span):
    try:
        ema = data.ewm(span=span, adjust=False).mean()
        return ema

    except Exception as e:
        logger.error(f"Error calc_ema: {str(e)}")


def __calc_macd(data, fast_span=12, slow_span=26, signal_span=9):
    try:
        ema_fast = __calc_ema(data, fast_span)
        ema_slow = __calc_ema(data, slow_span)
        macd = ema_fast - ema_slow
        signal = __calc_ema(macd, signal_span)
        hist = macd - signal
        return macd, signal, hist

    except Exception as e:
        logger.error(f"Error calc_macd: {str(e)}")


def __financial_statement_columns(df, stock_id):
    try:
        financial_df = pd.read_csv(f"{config.path["data_path"]}/financial_statement.csv")
        financial_df["stock_code"] = financial_df["stock_code"].astype(str)
        financial_df = financial_df[financial_df["stock_code"] == stock_id]
        financial_df["submission_date"] = pd.to_datetime(financial_df["submission_date"])
        financial_df["effective_date"] = pd.to_datetime(financial_df["effective_date"])

        financial_df_ignore_columns = ["stock_code", "year", "quarter", "effective_date"]
        financial_df_filtered_columns = [column for column in financial_df.columns if not (column in financial_df_ignore_columns)]

        df = pd.merge_asof(df, financial_df[financial_df_filtered_columns], left_on="Date", right_on="submission_date", direction="backward")
        df.drop(columns=["submission_date"], inplace=True)

        return df

    except Exception as e:
        logger.error(f"Error financial_statement_columns: {str(e)}")


def __future_columns(df):
    try:
        # Added stock indicator futures
        future_df = pd.read_csv(f"{config.path["data_path"]}/future.csv")
        future_df["date"] = pd.to_datetime(future_df["date"])
        future_df.rename(columns={"date": "Date"}, inplace=True)
        future_first_df = future_df.drop_duplicates(subset=["Date"], keep="first")
        df = pd.merge(df, future_first_df[["Date", "volume", "close", "spread_per"]], on="Date", how="left")
        df.rename(columns={"close": "Future_Close", "volumn": "Future_Volumn", "spread_per": "Future_Increase"}, inplace=True)
        df["Future_avg_5"] = df["Future_Close"].rolling(window=5, min_periods=1).mean()
        df["Future_avg_10"] = df["Future_Close"].rolling(window=10, min_periods=1).mean()
        df["Future_avg_5_Increase"] = df["Future_avg_5"].pct_change()
        df["Future_avg_10_Increase"] = df["Future_avg_10"].pct_change()

        return df

    except Exception as e:
        logger.error(f"Error future_columns: {str(e)}")


def __change_columns(df):
    try:
        # Add number of consecutive rising and falling days
        df["change"] = df["Close"].diff()
        df["Streak"] = 0

        streak = 0
        for i in range(1, len(df)):
            if df["change"].iloc[i] > 0:
                streak = streak + 1 if streak > 0 else 1
            elif df["change"].iloc[i] < 0:
                streak = streak - 1 if streak < 0 else -1
            else:
                streak = 0
            df["Streak"].iloc[i] = streak

        df.drop(columns=["change"], inplace=True)

        return df

    except Exception as e:
        logger.error(f"Error change_columns: {str(e)}")


def __volume_columns(df):
    try:
        # Calculate 5-day rolling average of "volume", shift it to get yesterday's average, calculate the percentage increase, and drop intermediate columns
        df["Volume_avg"] = df["Volume"].rolling(window=5, min_periods=1).mean()
        df["Volume_avg_yesterday"] = df["Volume_avg"].shift(1)
        df["Volume_Increase"] = ((df["Volume_avg"] - df["Volume_avg_yesterday"]) / df["Volume_avg_yesterday"]) * 100
        df.drop(columns=["Volume_avg", "Volume_avg_yesterday"], inplace=True)

        return df

    except Exception as e:
        logger.error(f"Error volume_columns: {str(e)}")


def __paper_MACD_columns(df):
    try:
        # Add technical analysis features
        ema_period = config.data_preprocessing["ema_period_list"]

        df["EMA_m"] = __calc_ema(df["Close"], ema_period[0])
        df["EMA_n"] = __calc_ema(df["Close"], ema_period[1])
        macd, signal, _ = __calc_macd(df["Close"], ema_period[0], ema_period[1], ema_period[2])
        df["MACD"] = macd
        df["Signal"] = signal

        df["Strategy_TEMA"] = 0
        df["Strategy_MACD"] = 0
        for i in range(len(df)):
            if df["MACD"].iloc[i] > df["Signal"].iloc[i]:
                df["Strategy_TEMA"].iloc[i] = 1
            elif df["MACD"].iloc[i] < df["Signal"].iloc[i]:
                df["Strategy_TEMA"].iloc[i] = 2

            if df["EMA_m"].iloc[i] > df["EMA_n"].iloc[i] and df["EMA_m"].iloc[i] > df["Signal"].iloc[i]:
                df["Strategy_MACD"].iloc[i] = 1
            elif df["EMA_m"].iloc[i] < df["EMA_n"].iloc[i] and df["EMA_m"].iloc[i] < df["Signal"].iloc[i]:
                df["Strategy_MACD"].iloc[i] = 2

        return df

    except Exception as e:
        logger.error(f"Error paper_MACD_columns: {str(e)}")


def __add_new_columns(df):
    try:
        avg_list = config.data_preprocessing["close_avg_list"]

        # Add moving average columns
        for avg in avg_list:
            df[f"avg{avg}"] = df["Close"].rolling(window=avg, min_periods=1).mean()

        # Add percentage difference between moving averages columns
        for idx1 in range(1, len(avg_list)):
            for idx2 in range(idx1):
                df[f"pd_avg{avg_list[idx1]}_avg{avg_list[idx2]}"] = (df[f"avg{avg_list[idx1]}"] - df[f"avg{avg_list[idx2]}"]) / df[f"avg{avg_list[idx1]}"] * 100

        df = __paper_MACD_columns(df)
        df = __volume_columns(df)
        df = __change_columns(df)
        df = __future_columns(df)

        return df

    except Exception as e:
        logger.error(f"Error add_new_columns: {str(e)}")


def __dividend_columns(df, stock_id):
    try:
        dividend = pd.read_csv(f"{config.path["data_path"]}/dividend.csv")
        df["dividend"] = 0
        if dividend["stock_code"].isin([stock_id]).sum() != 0:
            stock_dividend = dividend[dividend["stock_code"] == stock_id]
            stock_dividend["Date"] = pd.to_datetime(stock_dividend["Date"])
            stock_dividend.sort_values("Date", inplace=True)

            df["Date"] = pd.to_datetime(df["Date"])
            for date, type in stock_dividend[["Date", "type"]].values:
                running_date = date
                for i in range(7):
                    if df["Date"].isin([running_date]).sum() > 0:
                        if type == "權":
                            df.loc[df["Date"] == running_date, "dividend"] = 1
                        elif type == "息":
                            df.loc[df["Date"] == running_date, "dividend"] = 2
                        elif type == "權息" or type == "息權":
                            df.loc[df["Date"] == running_date, "dividend"] = 3

                    running_date = running_date - pd.Timedelta(days=1)
                    if running_date < pd.Timestamp("2010-1-1"):
                        break

        return df

    except Exception as e:
        logger.error(f"Error dividend_columns: {str(e)}")


def __stock_index_columns(df):
    try:
        stock_index_df = pd.read_csv(f"{config.path["data_path"]}/stock_index_info.csv")
        # Read and concatenate stock index data
        stock_index_list = stock_index_df["code"].to_list()
        for stock_index in stock_index_list:
            stock_index_data = pd.read_csv(f"{config.path["stock_history_path"]}/{stock_index}.csv")
            stock_index_data.rename(columns={"Close": stock_index}, inplace=True)
            df = pd.concat([df, stock_index_data[stock_index]], axis=1).reindex(df.index)
            df[stock_index] = df[stock_index].fillna(df[stock_index].mean())

    except Exception as e:
        logger.error(f"Error stock_index_columns: {str(e)}")


def __institutional_investors_columns(df, stock_id):
    try:
        institutional_df = pd.read_csv(f"{config.path["data_path"]}/institutional.csv")
        institutional_df["stock_code"] = institutional_df["stock_code"].astype(str)
        institutional_df = institutional_df[institutional_df["stock_code"] == stock_id]
        institutional_df["Date"] = pd.to_datetime(institutional_df["Date"])
        df["Date"] = pd.to_datetime(df["Date"])

        rename_dict = {
            "Date": "Date",
            "外資買進股數": "foreign_buy_volume",
            "外資賣出股數": "foreign_sell_volume",
            "外資買賣超股數": "foreign_net_volume",
            "投信買進股數": "investment_trust_buy_volume",
            "投信賣出股數": "investment_trust_sell_volume",
            "投信買賣超股數": "investment_trust_net_volume",
            "自營商買進股數": "dealer_buy_volume",
            "自營商賣出股數": "dealer_sell_volume",
            "自營商買賣超股數": "dealer_net_volume",
            "三大法人買賣超股數": "three_institutional_net_volume",
        }
        institutional_df.rename(columns=rename_dict, inplace=True)
        institutional_columns = list(rename_dict.values())
        df = pd.merge(df, institutional_df[institutional_columns], on="Date", how="left")
        for column in ["foreign_net_volume", "investment_trust_net_volume", "dealer_net_volume", "three_institutional_net_volume"]:
            df[f"{column}_trade_status"] = df[column].apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
            df[f"{column}_status_change"] = df[f"{column}_trade_status"] != df[f"{column}_trade_status"].shift()

            df["group_id"] = df[f"{column}_status_change"].cumsum()
            df[f"{column}_consecutive_days"] = df.groupby("group_id").cumcount() + 1

            df.drop(columns=["group_id"], inplace=True)

        institutional_columns.remove("Date")
        df["institutional_volume"] = df[institutional_columns].astype(np.float32).sum(axis=1)

        return df

    except Exception as e:
        logger.error(f"Error institutional_investors_columns: {str(e)}")


def __stock_index_columns(df):
    try:
        stock_index_df = pd.read_csv(f"{config.path["data_path"]}/stock_index_info.csv")
        avg_list = config.data_preprocessing["close_avg_list"]
        # Read and concatenate stock index data
        stock_index_list = stock_index_df["code"].to_list()
        for stock_index in stock_index_list:
            stock_index_data = pd.read_csv(f"{config.path["stock_history_path"]}/{stock_index}.csv")
            stock_index_data = stock_index_data.rename(columns={"Close": stock_index})
            df = pd.concat([df, stock_index_data[stock_index]], axis=1).reindex(df.index)
            df[stock_index] = df[stock_index].fillna(df[stock_index].mean())

            for avg in avg_list:
                df[f"avg{avg}_{stock_index}"] = df[stock_index].rolling(window=avg, min_periods=1).mean()

        return df

    except Exception as e:
        logger.error(f"Error stock_index_columns: {str(e)}")


def __read_data(stock_id):
    try:
        # Read the data from the CSV file
        df = pd.read_csv(f"{config.path["news_scored_path"]}/{stock_id}.csv")

        data_length = config.data_preprocessing["stock_length_check"] + config.training["test_data_length"]
        if len(df) < data_length:
            return

        df["Date"] = pd.to_datetime(df["Date"])
        df.sort_values("Date", inplace=True)

        df = df[-data_length:]

        df = __stock_index_columns(df)
        df = __dividend_columns(df, stock_id)
        df = __financial_statement_columns(df, stock_id)
        df = __institutional_investors_columns(df, stock_id)

        return df

    except Exception as e:
        logger.error(f"Error read_data: {str(e)}")


def __data_grouping(stock_id, df, type="Train"):
    try:
        history_window = config.data_preprocessing["history_window_length"]
        predict_window = config.data_preprocessing["predict_window_length"]

        dividend = pd.read_csv(f"{config.path["data_path"]}/dividend.csv")
        dividend = dividend[dividend["stock_code"] == stock_id]
        dividend["Date"] = pd.to_datetime(dividend["Date"])
        dividend.sort_values("Date", inplace=True)

        def get_new_data(i):
            stock_train_df = df.iloc[i - history_window : i]
            new_data = stock_train_df[columns].rolling(window=5).mean().iloc[4::5].reset_index(drop=True).values.flatten().tolist()
            return new_data

        # Check data type validity
        if not type in ["Train", "Test"]:
            return

        # Convert "Date" column to datetime format and sort dataframe by date
        df["Date"] = pd.to_datetime(df["Date"])
        df.sort_values(by="Date", inplace=True)
        df_copy = df.copy()

        # Fill NaN and inf values after percentage change calculation
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

        # Set "Date" column as index
        df.set_index("Date", inplace=True)
        df = df.astype(np.float32)

        data = []
        columns = df.columns

        # Generate training data
        if type == "Train":
            # The past 20(history_window) days are training data, and calculating the average of the next 20(predict_window) days is the target
            for i in range(history_window, len(df) - predict_window):
                new_data = get_new_data(i)

                stock_predict_df = df_copy.iloc[i : i + predict_window]
                avg = stock_predict_df["Close"].mean()  # Calculating the average "Close"

                if dividend[(dividend["Date"] >= stock_predict_df["Date"].iloc[0]) & (dividend["Date"] <= stock_predict_df["Date"].iloc[-1])].empty == False:
                    dividend_item = dividend[(dividend["Date"] >= stock_predict_df["Date"].iloc[0]) & (dividend["Date"] <= stock_predict_df["Date"].iloc[-1])]

                    avg1 = stock_predict_df[stock_predict_df["Date"] < dividend_item["Date"].iloc[0]]["Close"].mean()
                    avg2 = stock_predict_df[stock_predict_df["Date"] >= dividend_item["Date"].iloc[0]]["Close"].mean()
                    price1 = stock_predict_df["Close"].iloc[0]
                    price2 = float(dividend_item["price"].iloc[0])
                    length1 = len(stock_predict_df[stock_predict_df["Date"] < dividend_item["Date"].iloc[0]])
                    length2 = len(stock_predict_df[stock_predict_df["Date"] >= dividend_item["Date"].iloc[0]])

                    pct1 = (avg1 - price1) / price1
                    pct2 = (avg2 - price2) / price2

                    y_pct = ((pct1 * length1) + (pct2 * length2)) / (length1 + length2)
                else:
                    y_pct = (avg - stock_predict_df["Close"].iloc[0]) / stock_predict_df["Close"].iloc[0]  # Calculating increase precent for target

                y_trend = 1 if y_pct > 0 else 0

                new_data.append(y_pct)
                new_data.append(y_trend)
                data.append(new_data)

            data = np.nan_to_num(data, nan=0, posinf=0, neginf=0)

            # Standardize features using StandardScaler
            sc = StandardScaler()
            data[:, :-2] = sc.fit_transform(data[:, :-2])  # Last 2 columns is target
            pickle.dump(sc, open(f"{config.path["model_path"]}/{stock_id}_sc.pkl", "wb"))

        # Generate test data
        elif type == "Test":
            for i in range(len(df) - 30, len(df)):
                new_data = get_new_data(i)
                data.append(new_data)

        return data

    except Exception as e:
        logger.error(f"Error data_grouping: {str(e)}")


def __data_preprocessing(stock_id):
    try:
        if os.path.isfile(f"{config.path["training_data_path"]}/{stock_id}.npy"):
            return

        # Read stock history data
        stock_history = __read_data(stock_id)

        # Check if there are sufficient data points
        if stock_history is None:  # Approximately 4 years of data
            return

        # Add new columns to the dataframe
        stock_history = __add_new_columns(stock_history)

        # Group data and generate training data
        data = __data_grouping(stock_id, stock_history)

        # Check if there are sufficient data points for each class
        if len(np.where(data[:, -1] == 0)[0]) < 2 or len(np.where(data[:, -1] == 1)[0]) < 2:
            return

        # Save training data
        np.save(f"{config.path["training_data_path"]}/{stock_id}", data)

        return

    except Exception as e:
        logger.error(f"Error data_preprocessing: {str(e)}")


def training_data_build(stock_id_list=None):
    try:
        print("Training data build.")
        __data_preprocessing("1101")
        if stock_id_list is None:
            stock_id_list_df = pd.read_csv(f"{config.path["data_path"]}/stock_info.csv")
            stock_id_list = stock_id_list_df["公司代號"].astype(str)

        p = Pool(processes=6)
        tqdm_bar = tqdm(total=len(stock_id_list), smoothing=0)
        for result in p.imap(__data_preprocessing, stock_id_list):
            tqdm_bar.update(1)

    except Exception as e:
        logger.error(f"Error data_getting: {str(e)}")
