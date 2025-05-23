import kagglehub
import pandas as pd
import os
from transformers import pipeline

path = kagglehub.dataset_download(
    "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews"
)

data = pd.read_csv(os.path.join(path, "IMDB Dataset.csv"))

print(data)

sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
)

results = sentiment_analyzer(data["review"].iloc[:5].tolist())

print(results)
