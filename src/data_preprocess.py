import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def preprocess(df: pd.DataFrame):
    logging.info("Combining title and description fields...")

    df["text"] = df["title"].fillna("") + " " + df["description"].fillna("")

    logging.info("Vectorising text using TF-IDF...")
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(df["text"])
    y = df["label"]

    return X, y, vectorizer