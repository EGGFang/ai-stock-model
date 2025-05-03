from utils.logger import Logger
import utils.config as config

import requests
import os
import time
import pandas as pd
import numpy as np

logger = Logger().get_logger()


def __financial_statements_file_getting():
    try:
        # URLs for old and new financial statements
        url_dict_list = {
            "old": {
                "B_S": "https://mops.twse.com.tw/mops/web/ajax_t51sb12",
                "P_L": "https://mops.twse.com.tw/mops/web/ajax_t51sb13",
            },
            "new": {
                "B_S": "https://mops.twse.com.tw/mops/web/ajax_t163sb05",
                "P_L": "https://mops.twse.com.tw/mops/web/ajax_t163sb04",
                "CFS": "https://mops.twse.com.tw/mops/web/ajax_t163sb20",
            },
        }
        # Parameters for old and new requests
        params_list = {
            "old": {
                "encodeURIComponent": 1,
                "run": None,
                "step": 1,
                "firstin": "ture",
                "TYPEK": "sii",
                "year": "",
                "season": "",
            },
            "new": {
                "encodeURIComponent": 1,
                "step": 1,
                "firstin": 1,
                "off": 1,
                "TYPEK": "sii",
                "year": "",
                "season": "",
            },
        }
        url_list_name = ["B_S", "P_L", "CFS"]
        for url in url_list_name:
            year, season = 99, 1
            while True:
                # Check if the data file already exists
                if os.path.isfile(f"{config.path["financial_statement_path"]}/{url}_{year}_{season}.csv"):
                    season += 1
                    if season > 4:
                        year += 1
                        season = 1
                    continue
                elif url == "CFS" and year < 102:
                    year, season = 102, 1  # Adjust for CFS data
                    continue

                # Set parameters for request based on year
                params = params_list["old"] if year < 102 else params_list["new"]
                params["year"] = str(year)
                params["season"] = f"{season:02}"

                url_dict = url_dict_list["old"] if year < 102 else url_dict_list["new"]

                r = requests.post(url_dict[url], params)
                if "查詢無資料" in r.text:  # Check for no data response
                    break
                financial_df_list = pd.read_html(r.text)
                financial_df = []
                for i in range(len(financial_df_list)):
                    if "公司代號" in financial_df_list[i].columns:
                        pass
                    elif "公司 代號" in financial_df_list[i].columns:
                        financial_df_list[i].rename(columns={"公司 代號": "公司代號"}, inplace=True)
                    else:
                        continue
                    financial_df.append(financial_df_list[i])

                # Concatenate all DataFrames and save to CSV
                financial_df_concat = pd.concat(financial_df, axis=0, sort=False)
                financial_df_concat.to_csv(f"{config.path["financial_statement_path"]}/{url}_{year}_{season}.csv", index=False)

                season += 1
                if season > 4:
                    year += 1
                    season = 1

                time.sleep(40)

    except Exception as e:
        logger.error(f"Error fetching stock financial info data: {str(e)}")


def __financial_statements_file_processing():
    try:
        # Retrieve stock info from the database
        stock_info_df = pd.read_csv(f"{config.path["data_path"]}/stock_info.csv")
        stock_info = stock_info_df[["公司代號", "普通股每股面額"]]
        stock_code_list = [str(stock_info["公司代號"].iloc[i]) for i in range(len(stock_info))]
        stock_value_dict = {str(stock_info["公司代號"].iloc[i]): str(stock_info["普通股每股面額"].iloc[i]) for i in range(len(stock_info))}

        # Columns to drop from the DataFrame(have string)
        drop_columns_name = ["換算匯率", "換算匯率參考依據", "是否經當地會計師查核或核閱"]

        # Dictionary to rename columns for consistency
        # Different industries or years will have different column names
        # And these are the same meaning
        rename_dict = {
            "稅前淨利（淨損）": [
                "繼續營業單位稅前淨利(淨損)",
                "繼續營業單位稅前淨利",
                "繼續營業單位稅前合併淨利(淨損)",
                "繼續營業單位稅前淨利",
                "繼續營業單位稅前純益(純損)",
                "繼續營業單位稅前損益",
            ],
            "稅後淨利（淨損）": [
                "本期稅後淨利（淨損）",
                "本期淨利（淨損）",
                "繼續營業單位稅後淨利(淨損)",
                "繼續營業單位淨利(淨損)",
                "繼續營業單位稅後合併淨利(淨損)",
                "繼續營業單位稅後純益(純損)",
                "繼續營業單位稅後合併淨利(淨損)",
                "繼續營業單位本期淨利（淨損）",
                "繼續營業單位稅後淨利",
            ],
            "權益總計": ["股東權益總計", "股東權益合計", "權益總額"],
            "資產總計": ["資產合計", "資產總額"],
            "負債總計": ["負債合計", "負債總額"],
            "每股盈餘": ["基本每股盈餘", "每股稅後盈餘", "基本每股盈餘（元）"],
        }

        # Get the most recent year and quarter for data insertion
        if os.path.isfile(f"{config.path["data_path"]}/financial_statement.csv"):
            financial_statement_old_df = pd.read_csv(f"{config.path["data_path"]}/financial_statement.csv")
            year, quarter = financial_statement_old_df.sort_values(by=["year", "quarter"])[-1]
        else:
            year, quarter = 99, 1  # Default to year 99 and season 1
            financial_statement_old_df = pd.DataFrame()

        financial_statement = []
        while True:
            # Initialize a sample DataFrame for calculations
            # Some of the numbers in the report are statistics for that year, so in order to subtract the data from the previous quarter, they need to be stored
            # Since this is an annual statistics, the data will be cleared in the first quarter of each year
            # "sample" is a sample project that can be directly copied for each stock
            # Because the number of shares is not necessarily exactly the same every year or quarter
            if quarter == 1:
                fs_df_season_dict = {
                    "sample": {
                        "營業收入": 0,
                        "營業成本": 0,
                        "營業費用": 0,
                        "稅前淨利（淨損）": 0,
                        "稅後淨利（淨損）": 0,
                        "營業活動之淨現金流入（流出）": 0,
                        "投資活動之淨現金流入（流出）": 0,
                        "籌資活動之淨現金流入（流出）": 0,
                    }
                }
            B_S_df = P_L_df = CFS_df = pd.DataFrame()

            # Load financial statement data from CSV files
            if os.path.isfile(f"{config.path["financial_statement_path"]}/B_S_{year}_{quarter}.csv"):
                B_S_df = pd.read_csv(f"{config.path["financial_statement_path"]}/B_S_{year}_{quarter}.csv")
                B_S_df.columns = B_S_df.columns.str.replace(" ", "").str.replace("\n", "")
                P_L_df = pd.read_csv(f"{config.path["financial_statement_path"]}/P_L_{year}_{quarter}.csv")
                P_L_df.columns = P_L_df.columns.str.replace(" ", "").str.replace("\n", "")
                if year >= 102:
                    CFS_df = pd.read_csv(f"{config.path["financial_statement_path"]}/CFS_{year}_{quarter}.csv")
                    CFS_df.columns = CFS_df.columns.str.replace(" ", "").str.replace("\n", "")

            else:
                break  # Exit if no data file found

            # Combine DataFrames for balance sheet, profit and loss, and cash flow statement
            fs_df = pd.concat([B_S_df, P_L_df, CFS_df], axis=1)

            # Remove field names that contain spaces or line breaks, and delete duplicate fields
            fs_df.columns = fs_df.columns.str.replace(" ", "").str.replace("\n", "")
            fs_df.columns = fs_df.columns.str.replace(r"\('(\w+)','\1'\)", r"\1", regex=True)
            fs_df = fs_df.loc[:, ~fs_df.columns.duplicated()]

            # Delete previously set fields that need to be deleted(bc these column inculde string values)
            for column in drop_columns_name:
                if column in fs_df.columns.to_list():
                    fs_df = fs_df.drop(columns=[column])

            # Rename columns in the DataFrame for consistency
            for convert_column in list(rename_dict.keys()):
                # If the field does not exist, add a new one
                if not (convert_column in fs_df.columns.to_list()):
                    fs_df[convert_column] = None

                # If a field with another name exists, go to the field with the specified name
                for rename_column in rename_dict[convert_column]:
                    if rename_column in fs_df.columns.to_list():
                        fs_df[convert_column] = fs_df[convert_column].fillna(fs_df[rename_column])

            # Convert nan to None because null in the database is None
            fs_df = fs_df.replace({np.nan, None})
            fs_df = fs_df.fillna("--")  # "--" mean no data

            # Check all type of nan convert to None
            for column in fs_df.columns.to_list():
                for i in range(len(fs_df)):
                    if pd.isna(fs_df[column].iloc[i]):
                        if not (fs_df[column].iloc[i] is None):
                            fs_df[column].iloc[i] = None
            fs_df = fs_df.replace("--", None)

            # There may be duplicate data table headers in the data table
            fs_df = fs_df[fs_df["公司代號"] != "公司代號"]
            fs_df = fs_df[fs_df["公司代號"] != "公司 代號"]

            for index, row in fs_df.iterrows():
                stock_code = str(row["公司代號"])
                # Make sure the stock symbol exists in the database
                if not (stock_code in stock_code_list):
                    continue

                # Determine whether stock information is recorded in dict
                if not (stock_code in list(fs_df_season_dict.keys())):
                    fs_df_season_dict[stock_code] = fs_df_season_dict["sample"]
                fs_df_season = fs_df_season_dict[stock_code]

                # Only use "NTD" as the base currency and delete unnecessary strings
                stock_value = stock_value_dict[stock_code]
                stock_value_pure = stock_value.replace("新台幣", "")
                stock_value_pure = stock_value_pure.replace(" ", "")
                stock_value_pure = stock_value_pure.replace("元", "")

                # Variable initialization
                financial_item = {
                    "operating_expenses": None,  # 營業費用
                    "operating_costs": None,  # 營業成本
                    "operating_revenue": None,  # 營業收入
                    "free_cash_flow": None,  # 自由現金流
                    "roe": None,  # 股東權益報酬率
                    "roa": None,  # 資產報酬率
                    "net_profit_margin": None,  # 淨利率
                    "opm": None,  # 營業利益率
                    "gross_margin": None,  # 毛利率
                    "operating_income": None,  # 營業利益
                    "gross_profit": None,  # 毛利
                    "income_before_tax": None,  # 稅前淨利
                    "income_after_tax": None,  # 稅後淨利
                    "equity": None,  # 權益總計
                    "assets": None,  # 資產總計
                    "liabilities": None,  # 負債總計
                    "bvps": None,  # 每股淨值
                    "eps": None,  # 每股盈餘
                    "current_ratio": None,  # 流動比率
                    "debt_ratio": None,  # 負債比率
                    "current_liabilities": None,  # 流動負債
                    "non_current_liabilities": None,  # 非流動負債
                    "current_assets": None,  # 流動資產
                    "non_current_assets": None,  # 非流動資產
                    "share_capital": None,  # 股本
                    "operating_activities_cash": None,  # 營業活動之現金流入
                    "investing_activities_cash": None,  # 投資活動之現金流入
                    "financing_activities_cash": None,  # 籌資活動之現金流入
                }

                if "營業收入" in fs_df.columns.to_list() and not (row["營業收入"] is None):
                    financial_item["operating_revenue"] = int(row["營業收入"]) - fs_df_season["營業收入"]
                    fs_df_season["營業收入"] = int(row["營業收入"])

                if "營業成本" in fs_df.columns.to_list() and not (row["營業成本"] is None):
                    financial_item["operating_costs"] = int(row["營業成本"]) - fs_df_season["營業成本"]
                    fs_df_season["營業成本"] = int(row["營業成本"])

                if "營業費用" in fs_df.columns.to_list() and not (row["營業費用"] is None):
                    financial_item["operating_expenses"] = int(row["營業費用"]) - fs_df_season["營業費用"]
                    fs_df_season["營業費用"] = int(row["營業費用"])

                if "稅前淨利（淨損）" in fs_df.columns.to_list() and not (row["稅前淨利（淨損）"] is None):
                    financial_item["income_before_tax"] = int(row["稅前淨利（淨損）"]) - fs_df_season["稅前淨利（淨損）"]
                    fs_df_season["稅前淨利（淨損）"] = int(row["稅前淨利（淨損）"])

                if "稅後淨利（淨損）" in fs_df.columns.to_list() and not (row["稅後淨利（淨損）"] is None):
                    financial_item["income_after_tax"] = int(row["稅後淨利（淨損）"]) - fs_df_season["稅後淨利（淨損）"]
                    fs_df_season["稅後淨利（淨損）"] = int(row["稅後淨利（淨損）"])

                if "股本" in fs_df.columns.to_list() and not (row["股本"] is None):
                    financial_item["share_capital"] = int(row["股本"])

                if "流動資產" in fs_df.columns.to_list() and not (row["流動資產"] is None):
                    financial_item["current_assets"] = int(row["流動資產"])

                if "非流動資產" in fs_df.columns.to_list() and not (row["非流動資產"] is None):
                    financial_item["non_current_assets"] = int(row["非流動資產"])

                if "流動負債" in fs_df.columns.to_list() and not (row["流動負債"] is None):
                    financial_item["current_liabilities"] = int(row["流動負債"])

                if "非流動負債" in fs_df.columns.to_list() and not (row["非流動負債"] is None):
                    financial_item["non_current_liabilities"] = int(row["非流動負債"])

                if "權益總計" in fs_df.columns.to_list() and not (row["權益總計"] is None):
                    financial_item["equity"] = int(row["權益總計"])

                if "資產總計" in fs_df.columns.to_list() and not (row["資產總計"] is None):
                    financial_item["assets"] = int(row["資產總計"])

                if "負債總計" in fs_df.columns.to_list() and not (row["負債總計"] is None):
                    financial_item["liabilities"] = int(row["負債總計"])

                if "營業活動之淨現金流入（流出）" in fs_df.columns.to_list() and not (row["營業活動之淨現金流入（流出）"] is None):
                    financial_item["operating_activities_cash"] = int(row["營業活動之淨現金流入（流出）"]) - fs_df_season["營業活動之淨現金流入（流出）"]
                    fs_df_season["營業活動之淨現金流入（流出）"] = int(row["營業活動之淨現金流入（流出）"])

                if "投資活動之淨現金流入（流出）" in fs_df.columns.to_list() and not (row["投資活動之淨現金流入（流出）"] is None):
                    financial_item["investing_activities_cash"] = int(row["投資活動之淨現金流入（流出）"]) - fs_df_season["投資活動之淨現金流入（流出）"]
                    fs_df_season["投資活動之淨現金流入（流出）"] = int(row["投資活動之淨現金流入（流出）"])

                if "籌資活動之淨現金流入（流出）" in fs_df.columns.to_list() and not (row["籌資活動之淨現金流入（流出）"] is None):
                    financial_item["financing_activities_cash"] = int(row["籌資活動之淨現金流入（流出）"]) - fs_df_season["籌資活動之淨現金流入（流出）"]
                    fs_df_season["籌資活動之淨現金流入（流出）"] = int(row["籌資活動之淨現金流入（流出）"])

                # 每股淨值
                if "每股淨值" in fs_df.columns.to_list() and not (row["每股淨值"] is None):
                    financial_item["bvps"] = float(row["每股淨值"])
                elif not (financial_item["share_capital"] is None):
                    if "新台幣" in stock_value and not (financial_item["equity"] is None):
                        financial_item["bvps"] = financial_item["equity"] / (financial_item["share_capital"] / float(stock_value_pure))

                # 每股盈餘
                if "每股盈餘" in fs_df.columns.to_list() and not (row["每股盈餘"] is None):
                    financial_item["eps"] = float(row["每股盈餘"])
                elif all(not (x is None) for x in [financial_item["income_after_tax"], financial_item["share_capital"]]):
                    financial_item["eps"] = financial_item["income_after_tax"] / (financial_item["share_capital"] / float(stock_value_pure))

                # 營業利益
                if all(
                    not (x is None)
                    for x in [
                        financial_item["operating_revenue"],
                        financial_item["operating_costs"],
                        financial_item["operating_expenses"],
                    ]
                ):
                    financial_item["operating_income"] = (
                        financial_item["operating_revenue"] - financial_item["operating_costs"] - financial_item["operating_expenses"]
                    )

                # 毛利
                if all(not (x is None) for x in [financial_item["operating_revenue"], financial_item["operating_costs"]]):
                    financial_item["gross_profit"] = financial_item["operating_revenue"] - financial_item["operating_costs"]

                # 流動比率
                if all(not (x is None) for x in [financial_item["current_assets"], financial_item["current_liabilities"]]):
                    financial_item["current_ratio"] = financial_item["current_assets"] / financial_item["current_liabilities"]

                # 負債比率
                if all(not (x is None) for x in [financial_item["liabilities"], financial_item["assets"]]):
                    financial_item["debt_ratio"] = financial_item["liabilities"] / financial_item["assets"]

                # 毛利率
                if all(not (x is None) for x in [financial_item["operating_revenue"], financial_item["gross_profit"]]):
                    financial_item["gross_margin"] = financial_item["gross_profit"] / financial_item["operating_revenue"]

                # 營業利益率
                if all(not (x is None) for x in [financial_item["operating_revenue"], financial_item["operating_income"]]):
                    financial_item["opm"] = financial_item["operating_income"] / financial_item["operating_revenue"]

                # 淨利率
                if all(not (x is None) for x in [financial_item["operating_revenue"], financial_item["income_after_tax"]]):
                    financial_item["net_profit_margin"] = financial_item["income_after_tax"] / financial_item["operating_revenue"]

                # 資產報酬率
                if all(not (x is None) for x in [financial_item["assets"], financial_item["income_after_tax"]]):
                    financial_item["roa"] = financial_item["income_after_tax"] / financial_item["assets"]

                # 股東權益報酬率
                if all(not (x is None) for x in [financial_item["equity"], financial_item["income_after_tax"]]):
                    financial_item["roe"] = financial_item["income_after_tax"] / financial_item["equity"]

                # 自由現金流
                if all(
                    not (x is None)
                    for x in [
                        financial_item["operating_activities_cash"],
                        financial_item["investing_activities_cash"],
                        financial_item["financing_activities_cash"],
                    ]
                ):
                    financial_item["free_cash_flow"] = financial_item["operating_activities_cash"] + financial_item["investing_activities_cash"]

                financial_statement.append(
                    {
                        "stock_code": stock_code,
                        "year": year,
                        "quarter": quarter,
                        "總資產": financial_item["assets"],
                        "總負債": financial_item["liabilities"],
                        "每股盈餘": financial_item["eps"],
                        "流動比率": financial_item["current_ratio"],
                        "股本": financial_item["share_capital"],
                        "負債比率": financial_item["debt_ratio"],
                        "毛利率": financial_item["gross_margin"],
                        "營業利益率": financial_item["opm"],
                        "淨利率": financial_item["net_profit_margin"],
                        "資產報酬率": financial_item["roa"],
                        "股東權益": financial_item["equity"],
                        "自由現金流": financial_item["free_cash_flow"],
                        "每股淨值": financial_item["bvps"],
                        "股東權益報酬率": financial_item["roe"],
                        "流動資產": financial_item["current_assets"],
                        "非流動資產": financial_item["non_current_assets"],
                        "流動負債": financial_item["current_liabilities"],
                        "非流動負債": financial_item["non_current_liabilities"],
                        "營業收入": financial_item["operating_revenue"],
                        "營業成本": financial_item["operating_costs"],
                        "毛利": financial_item["gross_profit"],
                        "營業費用": financial_item["operating_expenses"],
                        "營業利益": financial_item["operating_income"],
                        "稅前淨利": financial_item["income_before_tax"],
                        "稅後淨利": financial_item["income_after_tax"],
                        "經營活動現金流": financial_item["operating_activities_cash"],
                        "投資活動現金流": financial_item["investing_activities_cash"],
                        "融資活動現金流": financial_item["financing_activities_cash"],
                    }
                )

            quarter += 1
            if quarter > 4:
                year += 1
                quarter = 1

        financial_statement_new_df = pd.DataFrame(financial_statement)
        financial_statement_df = pd.concat([financial_statement_old_df, financial_statement_new_df])
        financial_statement_df.drop_duplicates(subset=["stock_code", "year", "quarter"], inplace=True)
        financial_statement_df.sort_values(["year", "quarter"], inplace=True)
        financial_statement_df.to_csv(f"{config.path["data_path"]}/financial_statement.csv", index=False)

    except Exception as e:
        logger.error(f"Error processing stock financial info data: {str(e)}")


def financial_statements_getting():
    print("Stock financial statements getting.")
    __financial_statements_file_getting()
    print("Stock financial statements processing.")
    __financial_statements_file_processing()
