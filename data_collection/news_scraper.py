from utils.logger import Logger
from utils.driver_manager import DriverManager
import utils.config as config

from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import random
import requests
from bs4 import BeautifulSoup
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm
import time
import re
import os

logger = Logger().get_logger()


def __convert_date_format(date_str):
    try:
        # Convert date string from "YYYY年MM月DD日" to "YYYY/MM/DD" format
        match = re.match(r"(\d+)年(\d+)月(\d+)日", date_str)
        year = match.group(1)
        month = match.group(2)
        day = match.group(3)
        return f"{year}/{month}/{day}"

    except Exception as e:
        logger.error(f"Error convert_date_format: {str(e)}")


def __yahoo_news_detail(url):
    try:
        check = 0
        while True:
            # Get HTML content from the URL
            r = requests.get(url)
            soup = BeautifulSoup(r.text, "html.parser")

            # Extract datetime and company information from the HTML
            if soup.find("time") == None:
                if check < 3:
                    check += 1
                    time.sleep(random.randint(4, 8))
                    continue
                else:
                    datetime = None
                    company = None

            else:
                datetime = __convert_date_format(soup.find("time").text)
                company = soup.find("span", class_="caas-author-byline-collapse").text

            return company, datetime

    except Exception as e:
        logger.error(f"Error yahoo_news_detail: {str(e)}")


def __get_yahoo_news(stock_id):
    try:
        # Check if there is an existing file for the stock's news
        # Process news until end_date
        old_df = pd.DataFrame()
        if os.path.isfile(f"{config.path["news_text_path"]}/yahoo{stock_id}.csv"):
            old_df = pd.read_csv(f"{config.path["news_text_path"]}/yahoo{stock_id}.csv")
            old_df["timestamp"] = pd.to_datetime(old_df["timestamp"])
            old_df.sort_values(by=["timestamp"], inplace=True)
            end_date = old_df["timestamp"].iloc[-1]
            end_date = pd.Timestamp(end_date)
        else:
            end_date = pd.Timestamp("2010-1-1")

        check_stock = 0
        while check_stock < 2:
            try:
                driver = DriverManager.get_driver()

                # Visit Yahoo Finance news page for the stock
                urls = f"https://tw.stock.yahoo.com/quote/{stock_id}.TW/news"
                check = 0
                while check < 3:
                    try:
                        driver.get(urls)
                        _ = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.XPATH, "//*[@id='YDC-Stream']/ul/li[1]/div/div/div[1]/h3")))
                        break
                    except:
                        check += 1

                # Scroll down to load more news articles
                move_height_time = 0
                while True:
                    last_height = driver.execute_script("return document.body.scrollHeight")
                    for i in range(10):
                        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                        time.sleep(0.2)
                    time.sleep(1)
                    new_height = driver.execute_script("return document.body.scrollHeight")

                    move_height_time += 1
                    if move_height_time == 50 or new_height == last_height:
                        break

                # Extract HTML content
                html = driver.page_source

                driver.quit()

                # Parse HTML content
                soup = BeautifulSoup(html, "html.parser")
                target_divs = soup.find_all("div", class_=["Ov(h) Pend(44px) Pstart(25px)", "Ov(h) Pend(14%) Pend(44px)--sm1024"])

                # Initialize lists to store news data
                titles = []
                summarys = []
                urls = []
                authors = []
                timestamps = []
                companys = []

                # Iterate through each news article
                for div in target_divs:
                    a_text = div.find("a").text if div.find("a") else None
                    a_url = div.find("a").attrs["href"] if div.find("a") else None
                    p_text = div.find("p").text if div.find("p") else None

                    span_text = div.find_all("span")

                    company, datetime = __yahoo_news_detail(a_url)
                    if end_date > pd.Timestamp(datetime):
                        break

                    titles.append(a_text)
                    urls.append(a_url)
                    summarys.append(p_text)
                    authors.append(span_text[0].text)
                    timestamps.append(datetime)
                    companys.append(company)

                # Check if any news articles are found
                if len(titles) == 0:
                    raise TypeError(f"{stock_id} no News.")

                # Create a DataFrame with the extracted data
                new_df = pd.DataFrame(
                    {
                        "title": titles,
                        "summary": summarys,
                        "url": urls,
                        "author": authors,
                        "company": companys,
                        "timestamp": timestamps,
                        "title_eng": None,
                        "summary_eng": None,
                    }
                )

                # Concatenate new data with existing data and drop duplicate URLs
                df = pd.concat([new_df, old_df])
                df = df.drop_duplicates(subset=["url"])
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df.sort_values(by=["timestamp"], inplace=True)

                # Save the DataFrame to a CSV file
                df.to_csv(f"{config.path["news_text_path"]}/yahoo{stock_id}.csv", index=False)
                break
            except:
                # Retry if there is an error
                time.sleep(random.randint(5, 20))
                driver.quit()
                check_stock += 1

        return

    except Exception as e:
        logger.error(f"Error get_yahoo_news: {str(e)}")


def __check_na(stock_id, get_news=False):
    try:
        # Check if the file exists
        if os.path.isfile(f"{config.path["news_text_path"]}/yahoo{stock_id}.csv") == False:
            print(f"yahoo{stock_id}.csv not exist.")
            return

        # Read the CSV file
        df = pd.read_csv(f"{config.path["news_text_path"]}/yahoo{stock_id}.csv")

        # Check if there are any NaN values
        if df["timestamp"].isna().sum() > 0:
            # If specified, attempt to retrieve missing timestamps from news articles
            if get_news:
                nan_timestamp_indices = df[df["timestamp"].isna()].index
                for idx in nan_timestamp_indices:
                    url = df.at[idx, "url"]
                    company, timestamp = __yahoo_news_detail(url)
                    if company is None:
                        df.drop([idx], inplace=True)
                    else:
                        df.at[idx, "company"] = company
                        df.at[idx, "timestamp"] = timestamp

                df.to_csv(f"{config.path["news_text_path"]}/yahoo{stock_id}.csv", index=False)
                print(f"yahoo{stock_id}.csv have {df['timestamp'].isna().sum()} nan.")
            else:
                print(f"yahoo{stock_id}.csv have {df['timestamp'].isna().sum()} nan.")

    except Exception as e:
        logger.error(f"Error check_na: {str(e)}")


def __get_yuanta_news(stock_id):
    try:
        # https://jdata.yuanta.com.tw/z/zc/zcv/zcv_2885_2023-12-29_4.djhtm

        # Check if there is an existing file for the stock's news
        # Process news until end_date
        old_df = pd.DataFrame()
        end_date = pd.Timestamp("2010-1-1")
        if os.path.isfile(f"{config.path["news_text_path"]}/yuanta{stock_id}.csv"):
            old_df = pd.read_csv(f"{config.path["news_text_path"]}/yuanta{stock_id}.csv")
            old_df["timestamp"] = pd.to_datetime(old_df["timestamp"])
            old_df.sort_values(by=["timestamp"], inplace=True)
            end_date = old_df["timestamp"].iloc[-1]
            end_date = pd.Timestamp(end_date)

        # Get current timestamp
        date = pd.Timestamp.now()

        # Initialize lists to store news data
        titles = []
        urls = []
        timestamps = []

        # Iterate backwards in time until reaching the end date
        while date >= end_date:
            for i in range(1, 6):
                check = 0
                while check < 2:
                    try:
                        # Send HTTP request to the Yuanta website
                        r = requests.get(f"https://jdata.yuanta.com.tw/z/zc/zcv/zcv_{stock_id}_{date.year}-{date.month}-{date.day}_{i}.djhtm", timeout=10)
                        soup = BeautifulSoup(r.text, "html.parser")

                        # Extract news elements from the HTML
                        if soup.find_all("td", class_="t3t1"):
                            element_list = soup.find_all("td", class_="t3t1")
                            for j in range(0, len(element_list), 2):
                                d = element_list[j].text.split("/")
                                title = element_list[j + 1].text
                                url = f"https://jdata.yuanta.com.tw{element_list[j + 1].find('a').attrs['href']}"
                                if len(old_df) > 0 and "url" in old_df.columns.tolist():
                                    if url in old_df["url"].values:
                                        continue
                                timestamps.append(f"{int(d[0])+1911}/{d[1]}/{d[2]}")
                                titles.append(title)
                                urls.append(url)

                            break
                        else:
                            check += 1
                            time.sleep(30)
                    except:
                        check += 1
                        time.sleep(30)

            # Update the date to the last date in the retrieved news
            last_date = timestamps[-1].split("/")
            if pd.Timestamp(f"{last_date[0]}-{last_date[1]}-{last_date[2]}") == date:
                break
            else:
                date = pd.Timestamp(f"{last_date[0]}-{last_date[1]}-{last_date[2]}")

        if len(urls):
            # Create a DataFrame with the extracted data
            new_df = pd.DataFrame({"title": titles, "url": urls, "timestamp": timestamps, "title_eng": None})

            # Concatenate new data with existing data and drop duplicate URLs
            df = pd.concat([new_df, old_df])
            df.drop_duplicates(subset=["url"], inplace=True)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.sort_values(by=["timestamp"], inplace=True)

            drop_str_list = ["總表", "排行", "交易標的名單", "法說會"]
            for drop_str in drop_str_list:
                df = df[~df["title"].str.contains(drop_str)]

            # Save the DataFrame to a CSV file
            df.to_csv(f"{config.path["news_text_path"]}/yuanta{stock_id}.csv", index=False)

        return

    except Exception as e:
        logger.error(f"Error get_yuanta_news: {str(e)}")


def news_getting():
    try:
        stock_id_list_df = pd.read_csv(f"{config.path["data_path"]}/stock_info.csv")
        stock_id_list = stock_id_list_df["公司代號"].astype(str)
        # __get_yahoo_news("1101")
        __get_yuanta_news("1608")

        # yahoo
        print("Getting yahoo news.")
        yahoo_p = Pool(processes=4)
        tqdm_bar = tqdm(total=len(stock_id_list), smoothing=0)
        for result in yahoo_p.imap(__get_yahoo_news, stock_id_list):
            tqdm_bar.update(1)
        tqdm_bar.close()
        del yahoo_p

        for stock_id in stock_id_list:
            __check_na(stock_id, True)

        # yuanta
        print("Getting yuanta news.")
        yuanta_p = Pool(processes=4)
        tqdm_bar = tqdm(total=len(stock_id_list), smoothing=0)
        for result in yuanta_p.imap(__get_yuanta_news, stock_id_list):
            tqdm_bar.update(1)
        tqdm_bar.close()
        del yuanta_p

    except Exception as e:
        logger.error(f"Error news_getting: {str(e)}")
