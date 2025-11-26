import logging
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score,
    classification_report, confusion_matrix
)

logger = logging.getLogger(__name__)

class ModelEvaluator:
    def evaluate(self, model, X_val, y_val):
        logger.info("Evaluating model...")

        preds = model.predict(X_val)

        acc = accuracy_score(y_val, preds)
        prec = precision_score(y_val, preds, average="weighted", zero_division=0)
        rec = recall_score(y_val, preds, average="weighted", zero_division=0)
        f1 = f1_score(y_val, preds, average="weighted", zero_division=0)

        logger.info(f"Accuracy: {acc:.4f}")
        logger.info(f"Precision: {prec:.4f}")
        logger.info(f"Recall: {rec:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")

        logger.info("Classification Report:")
        logger.info("\n" + classification_report(y_val, preds, zero_division=0))

        logger.info("Confusion Matrix:")
        logger.info("\n" + str(confusion_matrix(y_val, preds)))
