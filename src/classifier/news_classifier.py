import pickle
import logging
from pathlib import Path
from src.model.mode_io import ModelIO

logger = logging.getLogger(__name__)


class NewsClassifier:
    def __init__(self, model_path: str = "models/trained_model.pkl"):
        self.model_path = Path(model_path)
        self.LABEL_MAP = {
            1: "World",
            2: "Sports",
            3: "Business",
            4: "Sci/Tech"
        }

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        logger.info(f"Loading model from {self.model_path}")

        model_io = ModelIO()
        self.model, self.vectorizer = model_io.load(self.model_path)

    def classify(self, title: str, description: str) -> str:
        text = f"{title}. {description}"
        vectorized = self.vectorizer.transform([text])
        numeric_pred = self.model.predict(vectorized)[0]
        return self.LABEL_MAP.get(numeric_pred, "Unknown")
