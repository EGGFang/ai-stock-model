# Stock Return Prediction Using Ensemble Learning and Multi-Source Market Information

This study aims to develop a model capable of predicting the future returns of individual stocks, helping investors identify potentially better-performing stocks from a large pool of options. The model integrates techniques from time series analysis, financial data processing, and natural language processing, taking into account a wide range of variables including stock prices, financial statements, global market indices, major news events, and ex-dividend information. In the data preprocessing stage, different types of data are standardized and undergo feature engineering, such as moving averages, growth rate calculations, and sentiment score quantification, to improve data quality and model interpretability. The core of the model involves training and comparing multiple machine learning algorithms, and further enhances performance through ensemble learning techniques (stacking), which combine the strengths of various base models to improve stability and generalization. The methods employed in this study are based on recent research from the past five years across fields such as economics, computer science, and time series forecasting. The overall framework emphasizes interpretability, robustness, and practical applicability, aiming to deliver strong predictive performance in a volatile stock market environment influenced by sentiment and external events.

## Overview

* Combines **quantitative financial features** with **qualitative news data**
* Incorporates **international index influences**
* Uses **sliding average smoothing** to reduce noise from short-term volatility
* Evaluates multiple ML models and applies **stacking ensemble**
* Tests **model performance stability** over time

## Methodology

* **Data Sources**:

  * TWSE (stock prices, financial reports, ex-dividend data, institutional investor trading)
  * TAIFEX (index futures)
  * Yahoo Finance and TWSE for news
  * Global indices (e.g., TSE, GSPC, SOX, HSI, N225)

* **Target**: Predict the **average return over the next 20 trading days**

* **ML Models**:

  * `Lasso`, `Ridge`, `ElasticNet`, `Random Forest`, `XGBoost`, `LightGBM`, `CatBoost`
  * Combined via **Stacking Ensemble** for improved accuracy

* **Feature Engineering**:

  * Calculates growth ratios and financial ratios (e.g., debt ratio, ROE, gross margin)
  * News sentiment analysis using `Flair`, `TextBlob`, `VADER`, and `Transformer`
  * Applies moving and sliding averages for smoothing
  * Normalizes data on a per-stock basis
  * Incorporates ex-dividend adjustments to account for Taiwan market sensitivity to high dividend yield stocks

## Key Results
* Top 5 predicted stocks yield **9.25%**
* Top 10: **5.32%**, Top 15: **3.91%**
* No significant performance decay was observed across different time windows

Calculate the top 5 predicted return stocks for each model run and generate a return distribution chart.
![image](https://github.com/user-attachments/assets/a62f5c59-02ff-4f4a-8cdc-8138f481494f)


## Project Structure

```
.
├── data_collection/        # Data crawling and collection scripts
├── data_preprocessing/     # Data preprocessing and feature engineering
├── model/                  # Model tuning, training, and validation
├── utils/                  # Helper functions
├── webdriver/              # Web crawler configuration and tools
├── StackingAveragedModels.py  # Stacking ensemble model implementation
├── main.py                 # Entry point of the project
└── README.md
```

## References
* \[1] Reinforcement Learning for Systematic FX Trading
* \[2] Reinforcement Learning in Stock Trading
* \[3] Predicting Stock Movement Direction with Machine Learning : an Extensive Study on S&P 500 Stocks
* \[4] Stock Prediction by Integrating Sentiment Scores of Financial News and MLP-Regressor: A Machine Learning Approach
* \[5] Technical analysis strategy optimization using a machine learning approach in stock market indices
* \[6] An Exploratory Study on the Complexity and Machine Learning Predictability of Stock Market Data
* \[7] Practical Algorithmic Trading Using State Representation Learning and Imitative Reinforcement Learning
* \[8] An Empirical Study of Machine Learning Algorithms for Stock Daily Trading Strategy
* \[9] An empirical methodology for developing stockmarket trading systems using artificial neural networks
* \[10] An Overview of Machine Learning, Deep Learning, and Reinforcement Learning-Based Techniques in Quantitative Finance: Recent Progress and Challenges
* \[11] Stock Market Forecasting with Different Input Indicators using Machine Learning and Deep Learning Techniques: A Review
* \[12] Financial applications of machine learning: A literature review
* \[13] Using News Articles to Predict Stock Price Movements
* \[14] Leveraging social media news to predict stock index movement using RNN-Boost
* \[15] Financial news predicts stock market volatility better than close price
* \[16] Stock Market Prediction Using Neural Networks through News on Online Social Networks
* \[17] Back to the future: an empirical investigation into the validity of stock index models over time
* \[18] Stock Prediction based on Bayesian-LSTM
* \[19] A systematic review of stock market prediction using machine learning and statistical techniques
* \[20] Market sentiment-aware deep reinforcement learning approach for stock portfolio allocation
* \[21] A Survey of Forex and Stock Price Prediction Using Deep Learning
* \[22] A parallel multi-module deep reinforcement learning algorithm for stock trading
* \[23] Harvesting social media sentiment analysis to enhance stock market prediction using deep learning
* \[24] A synchronous deep reinforcement learning model for automated multi-stock trading
* \[25] Deep reinforcement learning based trading agents: Risk curiosity driven learning for financial rules-based policy
* \[26] A Novel Trading Strategy Framework Based on Reinforcement Deep Learning for Financial Market Predictions

## Notes

* For research purposes only. Not intended for financial advice.
* News data is dependent on source tagging quality and completeness.

