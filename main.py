import data_collection
import data_processing
import model
from utils import file_delete


def train():
    data_collection.stock_code_getting()
    data_collection.stock_data_getting()
    data_collection.institutional_investors_getting()
    data_collection.dividend_getting()
    data_collection.financial_statements_getting()
    data_collection.news_getting()
    data_processing.financial_processing()
    data_processing.translate_news()
    data_processing.news_score_getting()

    file_delete.model_delete()
    file_delete.params_delete()
    file_delete.training_data_delete()
    data_processing.training_data_build()
    model.get_model()
    model.model_score_getting()


def predict():
    data_collection.stock_code_getting()
    data_collection.stock_data_getting()
    data_collection.institutional_investors_getting()
    data_collection.dividend_getting()
    data_collection.financial_statements_getting()
    data_collection.news_getting()
    data_processing.financial_processing()
    data_processing.translate_news()
    data_processing.news_score_getting()

    model.predict_getting(6)
    model.predict_score_getting()


def main():
    # data_collection.stock_code_getting()
    # data_collection.stock_data_getting()
    # data_collection.institutional_investors_getting()
    # data_collection.dividend_getting()
    # data_collection.financial_statements_getting()
    # data_collection.news_getting()
    # data_processing.financial_processing()
    # data_processing.translate_news()
    # data_processing.news_score_getting()

    # data_processing.training_data_build()
    # model.get_model(2)
    # model.model_scoring.model_score_getting()
    model.predict_getting()


if __name__ == "__main__":
    # main()
    predict()
    # train()
