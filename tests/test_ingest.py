# tests/test_ingest.py
import pandas as pd
from pathlib import Path
from src.ingest import load_raw_data, clean_data

def test_load_raw_data():
    df = load_raw_data("sh0416/ag_news")
    assert not df.empty

def test_clean_data():
    raw_df = load_raw_data("sh0416/ag_news")
    clean_df = clean_data(raw_df)
    assert list(clean_df.columns) == ["label", "title", "description"]
    assert clean_df["title"].dtype == object
    assert clean_df["description"].dtype == object
    assert clean_df["label"].dtype in [int, "int32", "int64"]