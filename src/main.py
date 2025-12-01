import logging
from sklearn.model_selection import train_test_split


from data.dataset_ingestor import DatasetIngestor
from src.data.dataset_loader import DatasetLoader
from data.text_preprocessor import TextPreprocessor
from src.model.trainer import ModelTrainer
from src.model.evaluator import ModelEvaluator
from src.model.mode_io import ModelIO
from interfaces.ag_news_cli_interface import ag_news_cli_interface



logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
RANDOM_STATE = 42

def main():

    #ingest
    ingestor = DatasetIngestor()
    clean_df = ingestor.run()
    
    # load from local optional
    loader = DatasetLoader(random_state=RANDOM_STATE)
    clean_df = loader.load("data/cleaned_dataset.csv")

    # Preprocess
    processor = TextPreprocessor()
    X, y = processor.preprocess(clean_df)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    
    # Train
    trainer = ModelTrainer()
    model = trainer.train(X_train, y_train)
    
    # Evaluate
    evaluator = ModelEvaluator()
    evaluator.evaluate(model, X_test, y_test)
    
    # Save model
    model_io = ModelIO()
    model_io.save(model, processor.vectorizer)
    

if __name__ == "__main__":
    main()