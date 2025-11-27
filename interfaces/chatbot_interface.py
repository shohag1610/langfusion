import json
import logging
from typing import Dict
import re

from transformers import pipeline
from classifier.news_classifier import NewsClassifier
from utils.clean_text import clean_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatbotInterface:

    def __init__(
        self,
        model_path: str = "models/trained_model.pkl",
        llm_model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    ):
        self.classifier = NewsClassifier(model_path)
        
        self._extractor_pipeline = pipeline("text2text-generation", model=llm_model_name)
        self.extractor_fn = self._llm_extractor
        
    def _llm_extractor(self, user_input: str) -> Dict[str, str]:
        cleaned_user_input = clean_text(user_input)
        return cleaned_user_input

if __name__ == "__main__":
    bot = ChatbotInterface()

    result = bot._llm_extractor("FCC mobile spam rule doesn't cover some SMS (MacCentral). MacCentral - A rule prohibiting mobile-phone spam adopted by the U.S. Federal Communications Commission (FCC) earlier this month doesn't prohibit phone-to-phone text messaging, but FCC officials believe the new rule, combined with a 13-year-old law, should protect U.S. mobile phone customers against unsolicited commercial e-mail.")
    print(result)
