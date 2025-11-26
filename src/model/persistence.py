import pickle
import logging

logger = logging.getLogger(__name__)

class ModelPersistence:
    def save(self, model, vectorizer, output_path="models/trained_model.pkl"):
        with open(output_path, "wb") as f:
            pickle.dump({"model": model, "vectorizer": vectorizer}, f)
        logger.info(f"Model saved to {output_path}")
