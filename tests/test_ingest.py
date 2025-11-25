# tests/test_ingest.py
import pandas as pd
from pathlib import Path
from src.ingest import load_raw_data, clean_data, save_data

def test_load_raw_data():
    df = load_raw_data("sh0416/ag_news")
    assert not df.empty



