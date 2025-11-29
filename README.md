# 📰 LangFusion
A hybrid NLP system that combines LLMs with classical ML to intelligently classify news — all accessible through an interactive Chatbot-style CLI interface.    

 🔹 LLM Extractor → structures user input into clean JSON   
 🔹 TF-IDF + Logistic Regression → efficient, high-accuracy news classification     
 🔹 Dynamic Model Persistence → auto-saves vectorizers, label maps & models     
 🔹 Chatbot CLI Interface → chat and classify directly from your terminal   
 🔹 Modular & Test-Driven Architecture → clean, extendable, production-ready    

## 📚 Dataset Overview

Name: AG News Classification Dataset
Source: HuggingFace
Description: Each sample contains a news title and description, categorized into one of four classes:

 🔹 World   
 🔹 Sports  
 🔹 Business    
 🔹 Sci/Tech    

Size:
Total: 120,000 samples  
Usage: Dataset is downloaded locally, preprocessed, and used to train the classifier.

## Project Structure
```bash
langfusion/
│
├── classifier/			        # news classifier (user local model)
│   └── news_classifier.py  
│  
├── data/			            # datasets
│   ├── cleaned_dataset.csv
│   └── raw_dataset.csv
│
├── interfaces/			       
│   ├── chatbot_interface.py
│   └── ag_news_chat_cli.py 
│  
├── models/			            # locally saved models (trained)
│   └── trained_model.pkl   
│
├── src/
│   ├── data/           	    # ingest pipeline, preprocessor, data loader
│   ├── model/          	    # model training, evaluation, saving and loading (locally)
│   └──  main.py 
│
├── tests/			
│   └── test_dataset_ingestor.py
│
├── utils/                      # helper functions
│   └──  clean_text.py 
│
├── venv/	
├── README.md
├── .gitignore
└── requirements.txt   
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
3. (Download + Clean + Split) the dataset + (Train + Save) the Model locally (run bellow from project root):
```bash
python src/main.py
```
## 💬 Interacting with the Local Model (CLI)
Get the CLI to interact with local model (run bellow from project root):
```bash
python interfaces/interactive_cli_interface.py
```
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

## 💬 Interacting with Chatbot (CLI)
Run the this to load Chatbot CLI (from project root):
```bash
python interfaces/chatbot_interface.py
```
You will see a prompt asking for a news details:    

You: [enter a news details] 

Assistant: [Few lines telling what type of news is this]   
  
### The model will respond with the predicted news category:
[Result] [What type of news is this]

To exit the interface, type:
```bash
exit
```