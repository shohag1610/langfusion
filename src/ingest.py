# ingest.py
import logging
from pathlib import Path
import pandas as pd
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_raw_data(dataset_name: str = "sh0416/ag_news", split: str = "train"):
    logging.info(f"Loading dataset from HuggingFace → {dataset_name}, split={split}")
    
    dataset = load_dataset(dataset_name, split=split)
    df = dataset.to_pandas()
    
    logging.info("Raw data loaded successfully.")
    
    return df




def ingest():
    raw_df = load_raw_data() 
    
    logging.info("Ingestion pipeline completed successfully.")


if __name__ == "__main__":
    ingest()
