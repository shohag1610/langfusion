import logging
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, max_iter=300):
        self.model = LogisticRegression(max_iter=max_iter)

    def train(self, X_train, y_train):
        logger.info("Training Logistic Regression model...")
        self.model.fit(X_train, y_train)
        return self.model
