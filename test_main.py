from main import *


def test_load_dataset():
    data = load_dataset()
    assert isinstance(
        data, pd.DataFrame
    ), "The returned dataset should be of type pd.DataFrame"
    assert len(data) > 0, "The returned dataset should contain items"


def test_load_sentiment_analyzer():
    analyzer = load_sentiment_analyzer(
        "Varnikasiva/sentiment-classification-bert-mini"
    )
    assert isinstance(
        analyzer, Pipeline
    ), "The returned analyzer should be of type Pipeline"
