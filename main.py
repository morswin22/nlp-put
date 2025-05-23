import kagglehub
import pandas as pd
import os
from transformers import pipeline
from transformers.pipelines import Pipeline


def load_dataset() -> pd.DataFrame:
    """Loads the IMDB movie reviews dataset from kaggle

    Returns:
        pd.DataFrame: contains review column and sentiment column with values "positive" and "negative"
    """

    path = kagglehub.dataset_download(
        "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews"
    )

    return pd.read_csv(os.path.join(path, "IMDB Dataset.csv"))


def load_sentiment_analyzer(model: str) -> Pipeline:
    """Loads sentiment analyzer using transformers pipeline

    Args:
        model (str): The name of the model to use for the sentiment analysis

    Returns:
        Pipeline: The pipeline capable of sentiment analysis
    """
    return pipeline("sentiment-analysis", model=model)


if __name__ == "__main__":
    data = load_dataset()
    print(data)

    sentiment_analyzer = load_sentiment_analyzer(
        "distilbert-base-uncased-finetuned-sst-2-english"
    )

    results = sentiment_analyzer(data["review"].iloc[:5].tolist())
    print(results)
