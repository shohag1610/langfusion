import json
import logging
from typing import Dict
import re

from transformers import pipeline
from classifier.news_classifier import NewsClassifier
from utils.clean_text import clean_text
from utils.extract_ag_new_json import extract_ag_news_json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatbotInterface:

    def __init__(
        self,
        model_path: str = "models/trained_model.pkl",
        llm_model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    ):
        self.classifier = NewsClassifier(model_path)
        
        self.extractor_pipeline = pipeline("text2text-generation", model=llm_model_name)
        
    def llm_extractor(self, user_input: str) -> Dict[str, str]:
        cleaned_user_input = clean_text(user_input)
        prompt = (
            "Extract a news title and a short description from the user's input. "
            "Return a JSON object with keys: title, description. Return NOTHING else.\n\n"
            f"User input:\n{cleaned_user_input}\n\nJSON: "
        )
        
        logger.debug("Calling LLM extractor pipeline.")
        
        out = self.extractor_pipeline(prompt, max_length=256, do_sample=False)[0]["generated_text"]
        
        return extract_ag_news_json(out)
    
    def extract_and_classify(self, user_input: str) -> Dict[str, str]:
    
        extracted = self.llm_extractor(user_input)
        title = extracted.get("title", "")
        description = extracted.get("description", "")

        # call classifier
        label = self.classifier.classify(title, description)
        return {"title": title, "description": description, "label": label}

if __name__ == "__main__":
    bot = ChatbotInterface()

    result = bot.extract_and_classify("Heiress Nicky Hilton Marries in Vegas, LAS VEGAS - Nicky Hilton, the hotel heiress and socialite, has tied the knot with her beau in a late-night ceremony, according to court filings obtained by The Associated Press.    Hilton, 20, married New York money manager Todd Andrew Meister, 33, at the Las Vegas Wedding Chapel early Sunday, according a Clark County marriage license...")
    print(result)
