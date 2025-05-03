from utils.logger import Logger
import utils.config as config

import pandas as pd
import datetime
import numpy as np

logger = Logger().get_logger()


def __quarter_convert_date(year, quarter):
    try:
        if quarter == 5:
            quarter = 1
            year += 1
        if quarter == 1:
            return datetime.datetime(year + 1911, 5, 15)
        elif quarter == 2:
            return datetime.datetime(year + 1911, 8, 31)
        elif quarter == 3:
            return datetime.datetime(year + 1911, 11, 14)
        elif quarter == 4:
            return datetime.datetime(year + 1911 + 1, 3, 31)

    except Exception as e:
        logger.error(f"Error quarter_convert_date: {str(e)}")


def __financial_add_columns():
    try:
        stock_info_df = pd.read_csv(f"{config.path["data_path"]}/stock_info.csv")
        stock_info_df.rename(columns={"公司代號": "stock_code", "產業別": "industry"}, inplace=True)

        financial_df = pd.read_csv(f"{config.path["data_path"]}/financial_statement.csv")
        financial_df_ignore_columns = ["stock_code", "year", "quarter", "submission_date", "effective_date", "industry"]

        if not ("submission_date" in financial_df.columns):
            financial_df["submission_date"] = None
            financial_df["effective_date"] = None
        for i in range(len(financial_df)):
            if pd.isna(financial_df["submission_date"].iloc[i]):
                financial_df["submission_date"].iloc[i] = __quarter_convert_date(financial_df["year"].iloc[i], financial_df["quarter"].iloc[i])
                financial_df["effective_date"].iloc[i] = __quarter_convert_date(financial_df["year"].iloc[i], financial_df["quarter"].iloc[i] + 1)

        financial_df["submission_date"] = pd.to_datetime(financial_df["submission_date"])
        financial_df["effective_date"] = pd.to_datetime(financial_df["effective_date"])
        financial_df.sort_values(by=["stock_code", "year", "quarter"], inplace=True)

        financial_df = pd.merge(financial_df, stock_info_df[["stock_code", "industry"]], on="stock_code", how="left")
        for column in financial_df.columns:
            if column in financial_df_ignore_columns:
                continue
            if "pct" in column or "growth" in column:
                continue

            financial_df[f"{column}_overall_rank_pct"] = financial_df.groupby(["year", "quarter"])[column].rank(pct=True).round(5)
            financial_df[f"{column}industry_rank_pct"] = financial_df.groupby(["year", "quarter", "industry"])[column].rank(pct=True).round(5)

            financial_df["quarter_growth"] = financial_df.groupby("stock_code")[column].pct_change().round(5)
            financial_df["semiannual_growth"] = financial_df.groupby("stock_code")[column].pct_change(periods=2).round(5)
            financial_df["annual_growth"] = financial_df.groupby("stock_code")[column].pct_change(periods=4).round(5)

            financial_df[f"{column}_quarter_growth_rank_pct"] = financial_df.groupby(["year", "quarter"])["quarter_growth"].rank(pct=True).round(5)
            financial_df[f"{column}_annual_growth_rank_pct"] = financial_df.groupby(["year"])["annual_growth"].rank(pct=True).round(5)
            financial_df[f"{column}_semiannual_growth_rank_pct"] = financial_df.groupby(["year", "quarter"])["semiannual_growth"].rank(pct=True).round(5)

            financial_df[f"{column}_quarter_growth_industry_rank_pct"] = (
                financial_df.groupby(["year", "quarter", "industry"])["quarter_growth"].rank(pct=True).round(5)
            )
            financial_df[f"{column}_annual_growth_industry_rank_pct"] = financial_df.groupby(["year", "industry"])["annual_growth"].rank(pct=True)
            financial_df[f"{column}_semiannual_growth_industry_rank_pct"] = (
                financial_df.groupby(["year", "quarter", "industry"])["semiannual_growth"].rank(pct=True).round(5)
            )

            financial_df.drop(columns=["quarter_growth", "semiannual_growth", "annual_growth"], inplace=True)

        financial_df.fillna(0, inplace=True)
        financial_df.drop(columns=["industry"], inplace=True)
        financial_df.to_csv(f"{config.path["data_path"]}/financial_statement.csv", index=False)

    except Exception as e:
        logger.error(f"Error financial_add_columns: {str(e)}")


def financial_processing():
    print("Financial processing")
    __financial_add_columns()
