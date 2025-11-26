import logging
import pandas as pd

logger = logging.getLogger(__name__)

class DatasetLoader:
    def __init__(self, shuffle=True, random_state=42):
        self.shuffle = shuffle
        self.random_state = random_state

    def load(self, path: str) -> pd.DataFrame:
        logger.info(f"Loading dataset from {path}")
        df = pd.read_csv(path)

        if self.shuffle:
            df = df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)

        return df
