import logging
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

class TextPreprocessor:
    def __init__(self, stop_words="english"):
        self.vectorizer = TfidfVectorizer(stop_words=stop_words)

    def preprocess(self, df: pd.DataFrame):
        logger.info("Combining title and description fields...")
        df["text"] = df["title"].fillna("") + " " + df["description"].fillna("")

        logger.info("Vectorising text using TF-IDF...")
        X = self.vectorizer.fit_transform(df["text"])
        y = df["label"]
        return X, y
