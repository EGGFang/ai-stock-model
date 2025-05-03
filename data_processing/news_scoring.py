from utils.logger import Logger
import utils.config as config

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import yake
import string
from collections import Counter
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

from flair.models import TextClassifier
from flair.data import Sentence
from transformers import pipeline
from tqdm import tqdm
from multiprocessing import Pool
import pandas as pd
import numpy as np
import pickle
import time
import os
import warnings

logger = Logger().get_logger()
warnings.filterwarnings("ignore")
nltk.download("vader_lexicon", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
analyzer = SentimentIntensityAnalyzer()
sentiment_model = TextClassifier.load("en-sentiment")
transformer_sentiment_model = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english", device="cuda:0")

plan = ["vender", "textblob", "transformer", "flair"]


def __dl_sentiment_score(news_text):
    try:
        # Tokenize the news text into sentences
        sent_text = nltk.sent_tokenize(news_text)

        # Initialize score dictionary with count of sentences
        score = {"count": len(sent_text)}

        # Iterate through each sentiment analysis plan
        for plan_name in plan:
            score[plan_name] = 0  # Initialize score for each plan

        if "transformer" in plan:
            for sent in sent_text:
                if len(sent) <= 500:
                    transformer_value_list = transformer_sentiment_model(sent)
                    for transformer_value in transformer_value_list:
                        if transformer_value["label"] == "POSITIVE":
                            transformer_score = transformer_value["score"]
                        else:
                            transformer_score = -transformer_value["score"]
                        score["transformer"] += transformer_score
        if "flair" in plan:
            sentences = [Sentence(text) for text in sent_text]
            sentiment_model.predict(sentences)
            for sentence in sentences:
                flair_value = sentence.labels[0].to_dict()["value"]
                if flair_value == "POSITIVE":
                    flair_score = sentence.to_dict()["labels"][0]["confidence"]
                else:
                    flair_score = -(sentence.to_dict()["labels"][0]["confidence"])
                score["flair"] += flair_score
        for text in sent_text:
            if "vender" in plan:
                vender_score = analyzer.polarity_scores(text)["compound"]  # [-1, 1]
                score["vender"] += vender_score
            if "textblob" in plan:
                testimonial = TextBlob(text)
                textblob_score = testimonial.sentiment.polarity  # [-1, 1]
                score["textblob"] += textblob_score

        return score

    except Exception as e:
        logger.error(f"Error dl_sentiment_score: {str(e)}")


def __news_scoring_dl_count(stock_history, news_df):
    try:
        # Convert end date to Timestamp
        stock_history["Date"] = pd.to_datetime(stock_history["Date"])
        stock_history.sort_values(by="Date", inplace=True)
        start_date = pd.Timestamp(stock_history["Date"].values[0])

        # If news date is before the start date, drop data
        news_df.dropna(inplace=True)
        news_df["timestamp"] = pd.to_datetime(news_df["timestamp"])
        news_df = news_df[news_df["timestamp"] >= start_date]
        # Iterate through each news article in the DataFrame
        for title, summary, date in news_df[["title_eng", "summary_eng", "timestamp"]].values:
            text = title + summary

            # Calculate sentiment score for the news article
            score = __dl_sentiment_score(text)

            # Iterate until a matching date is found in stock history or until reaching the end date
            while True:
                if date in stock_history["Date"].values:
                    # Update stock history with sentiment scores
                    for plan_name in plan:
                        stock_history.loc[stock_history["Date"] == date, plan_name] += score[plan_name]
                    stock_history.loc[stock_history["Date"] == date, "news_count"] += score["count"]
                    break
                else:
                    # Move to the previous day if the date is not found in stock history
                    date = date - pd.Timedelta(days=1)

                    # If the news date is before the end date, print a warning and stop processing
                    if date < start_date:
                        print(f"Cant find date({date})")
                        break

        return stock_history

    except Exception as e:
        logger.error(f"Error news_scoring_dl_count: {str(e)}")


def __news_score_dl_setting(stock_id):
    try:
        if not os.path.isfile(f"{config.path["stock_history_path"]}/{stock_id}.csv"):
            return

        # Read stock history data
        stock_history = pd.read_csv(f"{config.path["stock_history_path"]}/{stock_id}.csv")
        stock_history["Date"] = pd.to_datetime(stock_history["Date"])

        # Remove dates already scored if there is an existing scored file
        old_df = pd.DataFrame()
        if os.path.isfile(f"{config.path["news_scored_path"]}/{stock_id}.csv"):
            old_df = pd.read_csv(f"{config.path["news_scored_path"]}/{stock_id}.csv")
            old_df["Date"] = pd.to_datetime(old_df["Date"])
            stock_history = stock_history[~stock_history["Date"].isin(old_df["Date"])]

        # Initialize sentiment score columns and news count column
        for plan_name in plan:
            stock_history[plan_name] = 0
        stock_history["news_count"] = 0

        # Process news articles if there are any
        if len(stock_history) > 0:
            # Process translated Yahoo news articles if available
            if os.path.isfile(f"{config.path["news_text_path"]}/yahoo{stock_id}.csv"):
                news_df = pd.read_csv(f"{config.path["news_text_path"]}/yahoo{stock_id}.csv")
                stock_history = __news_scoring_dl_count(stock_history, news_df)

            # Process translated Yuanta news articles if available
            if os.path.isfile(f"{config.path["news_text_path"]}/yuanta{stock_id}.csv"):
                news_df = pd.read_csv(f"{config.path["news_text_path"]}/yuanta{stock_id}.csv")
                news_df["summary_eng"] = ""
                stock_history = __news_scoring_dl_count(stock_history, news_df)

            # Convert sentiment scores to averages if there are scored news articles
            stock_history.loc[stock_history["news_count"] == 0, plan] = np.nan
            if (stock_history["news_count"] > 0).sum() > 0:
                stock_history.loc[stock_history["news_count"] > 0, plan] = (
                    stock_history.loc[stock_history["news_count"] > 0, plan].div(stock_history["news_count"], axis=0).round(5)
                )

        # Concatenate new data with existing scored data and drop duplicate dates
        df = pd.concat([stock_history, old_df])
        df.drop_duplicates(subset=["Date"], inplace=True)
        df.sort_values("Date", inplace=True)
        df.fillna(0, inplace=True)
        # Save the scored data to a file
        df.to_csv(f"{config.path["news_scored_path"]}/{stock_id}.csv", index=False)

    except Exception as e:
        logger.error(f"Error news_score_dl_setting: {str(e)}")


def news_scoring_dl_model():
    try:
        stock_id_list_df = pd.read_csv(f"{config.path["data_path"]}/stock_info.csv")
        stock_id_list = stock_id_list_df["公司代號"].astype(str)
        print("Clac dl news score.")
        # __news_score_dl_setting("1102")
        news_score_p = Pool(processes=1)
        tqdm_bar = tqdm(total=len(stock_id_list), smoothing=0)
        for result in news_score_p.imap(__news_score_dl_setting, stock_id_list):
            tqdm_bar.update(1)
        tqdm_bar.close()

    except Exception as e:
        logger.error(f"Error news_scoring_dl_model: {str(e)}")


# ----------------------------------------------------------------------


def __increase_filter(stock_id, increase_value, limit_value):
    try:

        def __news_string_processing(path):
            news = pd.read_csv(path)
            news["timestamp"] = pd.to_datetime(news["timestamp"])
            news = news[news["timestamp"].isin(data["Date"])]
            if "summary" in news.columns.to_list():
                news["text"] = news["title_eng"] + " " + news["summary_eng"]
            else:
                news["text"] = news["title_eng"]
            news_string = news["text"].str.cat(sep=" ")
            return news_string

        data = pd.read_csv(f"{config.path["stock_history_path"]}/{stock_id}.csv")
        data["Date"] = pd.to_datetime(data["Date"])
        data.sort_values("Date", inplace=True)
        data.astype({"Close": np.float16})
        data["Increase"] = data["Close"].pct_change()
        data = data[data["Increase"] >= increase_value]
        data = data[data["Increase"] <= limit_value]
        data["Date"] = data["Date"] - pd.Timedelta(days=1)

        yahoo_path = f"{config.path["news_text_path"]}/yahoo{stock_id}.csv"
        yuanta_path = f"{config.path["news_text_path"]}/yuanta{stock_id}.csv"
        yahoo_news_string = __news_string_processing(yahoo_path) if os.path.isfile(yahoo_path) else ""
        yuanta_news_string = __news_string_processing(yuanta_path) if os.path.isfile(yuanta_path) else ""

        return yahoo_news_string, yuanta_news_string

    except Exception as e:
        logger.error(f"Error increase_filter: {str(e)}")


def __process_text(text):
    try:
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        words = word_tokenize(text)

        stop_words = set(stopwords.words("english"))
        filtered_words = [word for word in words if word not in stop_words]

        return filtered_words

    except Exception as e:
        logger.error(f"Error process_text: {str(e)}")


def __process_news(news):
    try:
        result_text = []
        for i in range(len(news)):
            text = news["title_eng"].iloc[i] + news["summary_eng"].iloc[i]
            processed_text = __process_text(text)
            result_text = result_text + processed_text

        return result_text

    except Exception as e:
        logger.error(f"Error process_news: {str(e)}")


def __extract_keywords(words):
    try:
        keywords_list = []
        for num in range(1, 3 + 1):
            custom_kw_extractor = yake.KeywordExtractor(lan="en", n=num, dedupLim=0.1, top=config.news_paper["keywords_count"], features=None)
            keywords = custom_kw_extractor.extract_keywords(words)
            for item in keywords:
                if item[0] not in keywords_list:
                    keywords_list.append(item[0])

        return keywords_list

    except Exception as e:
        logger.error(f"Error extract_keywords: {str(e)}")


def __clac_paper_news_score(stock_id):
    try:
        if not os.path.isfile(f"{config.path["news_scored_path"]}/{stock_id}.csv"):
            return

        with open(f"{config.path["news_path"]}/news_keywords.pkl", "rb") as f:
            keywords = pickle.load(f)
        pos_keywords, neg_keywords = keywords["pos_keywords"], keywords["neg_keywords"]

        stock_history = pd.read_csv(f"{config.path["news_scored_path"]}/{stock_id}.csv")
        stock_history["Date"] = pd.to_datetime(stock_history["Date"])
        stock_history.sort_values("Date", inplace=True)

        stock_history["paper_sentiment_score_sum"] = 0
        stock_history["paper_sentiment_score_count"] = 0
        if not ("paper_sentiment_score" in stock_history.columns):
            stock_history["paper_sentiment_score"] = None

        old_df = pd.read_csv(f"{config.path["news_scored_path"]}/{stock_id}.csv")
        old_df["Date"] = pd.to_datetime(old_df["Date"])
        old_df.sort_values("Date", inplace=True)

        # stock_history = stock_history[~stock_history["Date"].isin(old_df["Date"])]
        start_date = pd.Timestamp(stock_history["Date"].values[0])
        end_date = pd.Timestamp(stock_history["Date"].values[-1])

        yahoo_news = yuanta_news = pd.DataFrame(columns=["timestamp", "title_eng", "summary_eng"])
        if os.path.isfile(f"{config.path["news_text_path"]}/yahoo{stock_id}.csv"):
            yahoo_news = pd.read_csv(f"{config.path["news_text_path"]}/yahoo{stock_id}.csv")
            yahoo_news["timestamp"] = pd.to_datetime(yahoo_news["timestamp"])

        if os.path.isfile(f"{config.path["news_text_path"]}/yuanta{stock_id}.csv"):
            yuanta_news = pd.read_csv(f"{config.path["news_text_path"]}/yuanta{stock_id}.csv")
            yuanta_news["timestamp"] = pd.to_datetime(yuanta_news["timestamp"])
            yuanta_news["summary_eng"] = ""

        result = 0
        news_count = 0
        running_date = start_date
        while running_date <= end_date:
            if sum(running_date == stock_history["Date"]) > 0 and stock_history[stock_history["Date"] == running_date]["paper_sentiment_score"].iloc[0] == None:
                yahoo_news_date = yahoo_news[yahoo_news["timestamp"] == running_date]
                yuanta_news_date = yuanta_news[yuanta_news["timestamp"] == running_date]
                news_date = pd.concat(
                    [
                        yahoo_news_date[["title_eng", "summary_eng"]],
                        yuanta_news_date[["title_eng", "summary_eng"]],
                    ]
                )
                news_date.dropna(inplace=True)
                for i in range(len(news_date)):
                    text = news_date["title_eng"].iloc[i] + " " + news_date["summary_eng"].iloc[i]
                    text_list = __process_text(text)
                    word_counts = Counter(text_list)

                    pos_keyword_counts = [word_counts.get(word, 0) for word in pos_keywords]
                    neg_keyword_counts = [word_counts.get(word, 0) for word in neg_keywords]

                    pos_keywords_sum = len(pos_keywords)
                    neg_keywords_sum = len(neg_keywords)
                    pos_keywords_prob = [count / pos_keywords_sum if count > 0 else 1 / pos_keywords_sum for count in pos_keyword_counts]
                    neg_keywords_prob = [count / neg_keywords_sum if count > 0 else 1 / neg_keywords_sum for count in neg_keyword_counts]

                    if sum(pos_keyword_counts) + sum(neg_keyword_counts):
                        pos_score = pos_keywords_sum / (pos_keywords_sum + neg_keywords_sum)
                        neg_score = neg_keywords_sum / (pos_keywords_sum + neg_keywords_sum)

                        for keywords_prob in pos_keywords_prob:
                            pos_score *= keywords_prob

                        for keywords_prob in neg_keywords_prob:
                            neg_score *= keywords_prob

                        score = pos_score / (pos_score + neg_score)
                    else:
                        score = 0

                    result += score
                    news_count += 1

                date = running_date
                while True:
                    if stock_history["Date"].isin([date]).sum():
                        stock_history.loc[stock_history["Date"] == date, "paper_sentiment_score_sum"] += result
                        stock_history.loc[stock_history["Date"] == date, "paper_sentiment_score_count"] += news_count
                        result = 0
                        news_count = 0
                        break
                    else:
                        date = date - pd.Timedelta(days=1)

                    if date < start_date:
                        break
            running_date += pd.Timedelta(days=1)

        stock_history["paper_sentiment_score"] = stock_history["paper_sentiment_score_sum"] / stock_history["paper_sentiment_score_count"]
        stock_history.loc[stock_history["paper_sentiment_score_count"] == 0, "paper_sentiment_score"] = 0
        stock_history.drop(columns=["paper_sentiment_score_sum", "paper_sentiment_score_count"], inplace=True)
        stock_history.sort_values("Date", inplace=True)

        stock_history.to_csv(f"{config.path["news_scored_path"]}/{stock_id}.csv", index=False)

        return

    except Exception as e:
        logger.error(f"Error clac_paper_news_score: {str(e)}")


def news_scoring_paper():
    try:
        stock_id_list_df = pd.read_csv(f"{config.path["data_path"]}/stock_info.csv")
        stock_id_list = stock_id_list_df["公司代號"].astype(str)
        print("Clac paper news score.")
        if not os.path.isfile(f"{config.path["news_path"]}/news_keywords.pkl"):
            positive_news = negative_news = ""
            for stock_id in stock_id_list:
                pos_yahoo_news, pos_yuanta_news = __increase_filter(stock_id, 0.085, 0.1)
                neg_yahoo_news, neg_yuanta_news = __increase_filter(stock_id, -0.1, -0.085)
                positive_news = positive_news + " " + pos_yahoo_news + " " + pos_yuanta_news
                negative_news = negative_news + " " + neg_yahoo_news + " " + neg_yuanta_news

            pos_keywords_processed = __process_text(positive_news)
            negative_news_processed = __process_text(negative_news)
            pos_keywords_text = " ".join(pos_keywords_processed)
            neg_keywords_text = " ".join(negative_news_processed)
            pos_keywords = __extract_keywords(pos_keywords_text)
            neg_keywords = __extract_keywords(neg_keywords_text)

            keywords = {"pos_keywords": pos_keywords, "neg_keywords": neg_keywords}
            with open(f"{config.path["news_path"]}/news_keywords.pkl", "wb") as f:
                pickle.dump(keywords, f)

        # __clac_paper_news_score("1101")
        tqdm_bar = tqdm(total=len(stock_id_list), smoothing=0)
        p = Pool(processes=4)
        for result in p.imap(__clac_paper_news_score, stock_id_list):
            tqdm_bar.update(1)

        tqdm_bar.close()

    except Exception as e:
        logger.error(f"Error get_paper_news_score: {str(e)}")


def news_score_getting():
    news_scoring_dl_model()
    news_scoring_paper()
