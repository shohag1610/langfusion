import logging
from src.ingest.dataset_ingestor import DatasetIngestor
from src.data.dataset_loader import DatasetLoader
from src.preprocessing.text_preprocessor import TextPreprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():

    #ingest
    ingestor = DatasetIngestor()
    clean_df = ingestor.run()
    
    # load from local optional
    # loader = DatasetLoader()
    # df = loader.load(clean_df)

    # Preprocess
    processor = TextPreprocessor()
    X, y = processor.preprocess(clean_df)
    
if __name__ == "__main__":
    main()