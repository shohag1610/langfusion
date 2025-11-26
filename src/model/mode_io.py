import pickle
import logging

logger = logging.getLogger(__name__)

class ModelIO:

    def save(self, model, vectorizer, output_path="models/trained_model.pkl"):
        payload = {
            "model": model,
            "vectorizer": vectorizer
        }

        with open(output_path, "wb") as f:
            pickle.dump(payload, f)

        logger.info(f"Model saved to {output_path}")

    def load(self, model_path="models/trained_model.pkl"):
        with open(model_path, "rb") as f:
            data = pickle.load(f)

        model = data.get("model")
        vectorizer = data.get("vectorizer")

        logger.info(f"Model loaded from {model_path}")
        return model, vectorizer
