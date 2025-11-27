import json
import logging
from typing import Callable, Dict, Optional

from transformers import pipeline
from classifier.news_classifier import NewsClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatbotInterface:

    def __init__(
        self,
        model_path: str = "models/trained_model.pkl",
        extractor_fn: Optional[Callable[[str], Dict[str, str]]] = None,
        responder_fn: Optional[Callable[[str, str, str], str]] = None,
        llm_model_name: str = "gpt2",
    ):
        self.classifier = NewsClassifier(model_path)
        self.extractor_fn = extractor_fn
        self.responder_fn = responder_fn
        