import logging
from pathlib import Path
import pandas as pd
from datasets import load_dataset

logger = logging.getLogger(__name__)

class DatasetIngestor:
    def __init__(self, output_dir="data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def load_raw_data(self, dataset_name: str = "sh0416/ag_news", split: str = "train") -> pd.DataFrame:
        logger.info(f"Loading dataset from HuggingFace  {dataset_name}, split={split}")
        dataset = load_dataset(dataset_name, split=split)
        df = dataset.to_pandas()
        logger.info("Raw data loaded successfully.")
        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame: 
        logger.info("Cleaning dataset...")

        rename_map = {
            "Label": "label",
            "Title": "title",
            "Description": "description"
        }

        df = df.rename(columns=rename_map)
        df["label"] = df["label"].astype(int)
        df["title"] = df["title"].astype(str).str.strip()
        df["description"] = df["description"].astype(str).str.strip()

        logger.info("Data cleaned successfully.")
        return df

    def save_data(self, df: pd.DataFrame, filename: str):
        output_path = self.output_dir / filename
        df.to_csv(output_path, index=False)
        logger.info(f"Dataset saved → {output_path}")

    def run(self):
        raw_df = self.load_raw_data()
        self.save_data(raw_df, "raw_dataset.csv")

        clean_df = self.clean_data(raw_df)
        self.save_data(clean_df, "cleaned_dataset.csv")

        logger.info("Ingestion pipeline completed successfully.")
        return clean_df
