# 📰 Natural Language Classifier — AG News
This project implements an end-to-end machine learning pipeline for classifying news headlines using the AG News dataset.
It combines traditional ML techniques with a structured workflow suitable for production-ready environments.

## 📚 Dataset Overview

Name: AG News Classification Dataset
Source: HuggingFace
Description: Each sample contains a news title and description, categorized into one of four classes:

World   
Sports  
Business    
Sci/Tech    

Size:
Total: 120,000 samples  
Usage: Dataset is downloaded locally, preprocessed, and used to train the classifier.

## Project Structure
```bash
langfusion/
│
├── classifier/			        # trained models
│   └── news_classifier.py  
│  
├── data/			        # datasets
│   ├── cleaned_dataset.csv
│   └── raw_dataset.csv
│  
├── models/			        # trained models
│   └── trained_model.pkl  
│
├── interfaces/			        # trained models
│   └── ag_news_chat_cli.py  
│
├── src/
│   ├── classifier/     	# news classifier
│   ├── data/           	# ingest pipeline, preprocessor, data loader
│   ├── model/          	# model training, evaluation, saving and loading, interaction-cli
│   └──  main.py 
│
├── tests/			
│   └── test_dataset_ingestor.py
│
├── requirements.txt		
├── README.md
├── .gitignore
└── venv/   
```

## ⚙️ How to Run the Project

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows

export PYTHONPATH=$(pwd)   # Set project root
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Train the Model (from project root):
```bash
python src/main.py
```
This downloads the dataset, trains the model, and saves it locally.

4. Run the Interactive CLI (from project root):
```bash
python interfaces/interactive_cli_interface.py
```
## 💬 Interacting with the Model (CLI)
You will see a prompt asking for a news title and description:

Enter news title: Sri Lanka hit by oil strike
Enter news description: Workers at Sri Lanka's main oil company end a two-day strike, held in protest at government plans to sell more of the company.

[Result] Business

### After this, continue with:
Enter news title: [Enter news title]    
Enter news description: [Enter description]     

### The model will respond with the predicted news category:
[Result] [What type of news is this]

To exit the interface, type:
```bash
exit
```