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

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Cleaning dataset...")

    rename_map = {
        "Label": "label",
        "Title": "title",
        "Description": "description"
    }

    df = df.rename(columns=rename_map)

    df["label"] = df["label"].astype(int)
    df["title"] = df["title"].astype(str).str.strip()
    df["description"] = df["description"].astype(str).str.strip()

    logging.info("Data cleaned successfully.")
    
    return df

def save_data(df: pd.DataFrame, filename: str):
    output_path = OUTPUT_DIR / filename
    df.to_csv(output_path, index=False)
    
    logging.info(f"Dataset saved → {output_path}")

def ingest():
    #get and save dataset from url
    raw_df = load_raw_data() 
    save_data(raw_df, "raw_dataset.csv")
    
    #clean data and save
    clean_df = clean_data(raw_df)
    save_data(clean_df, "cleaned_dataset.csv")
    
    logging.info("Ingestion pipeline completed successfully.")


if __name__ == "__main__":
    ingest()
