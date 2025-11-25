# Natural Language Classifier - AG News

This project builds a machine learning pipeline for classifying news headlines using the AG News dataset.
It combines a classifier with a language model, structured for professional, production-ready use.

## Dataset
- **Dataset Name:** AG News Classification Dataset
- **Source:** [HuggingFace](https://huggingface.co/datasets/sh0416/ag_news)
- **Description:** News headlines labeled into four categories: World, Sports, Business, Sci/Tech.
- **Size:** 120,000 training samples, 7,600 test samples
- **Usage:** Dataset downloaded and ingested locally for preprocessing and training

## How to Run

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
export PYTHONPATH=$(pwd)  # Export Current Working Directory
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Install dependencies: (make sure you are in root directory)
```bash
python src/ingest.py
```