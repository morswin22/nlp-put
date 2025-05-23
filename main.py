import kagglehub
import pandas as pd
import os

path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")

data = pd.read_csv(os.path.join(path, "IMDB Dataset.csv"))

print(data)
