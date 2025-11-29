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
        self.responder_pipeline = pipeline("text2text-generation", model='google/flan-t5-base')
        
    def llm_extractor(self, user_input: str) -> Dict[str, str]:
        cleaned_user_input = clean_text(user_input)
        prompt = (
            "Extract a news title and a short description from the user's input. "
            "Return a JSON object with keys: title, description. Return NOTHING else.\n\n"
            f"User input:\n{cleaned_user_input}\n\nJSON: "
        )
        
        logger.debug("Calling LLM extractor pipeline.")
        
        out = self.extractor_pipeline(prompt, max_length=1000, do_sample=False)[0]["generated_text"]
        
        return extract_ag_news_json(out)
    
    def extract_and_classify(self, user_input: str) -> Dict[str, str]:
    
        extracted = self.llm_extractor(user_input)
        title = extracted.get("title", "")
        description = extracted.get("description", "")

        # call classifier
        label = self.classifier.classify(title, description)
        return {"title": title, "description": description, "label": label}
    
    def llm_responder(self, user_input) -> str:
        
        classified_result = self.extract_and_classify(user_input)
        extracted_text = f"{classified_result['title']}. {classified_result['description']}".strip()
        classification = classified_result['label']
        
        prompt = f"""
            You are an assistant that explains ONLY the classification label.

            Task:
            Given the classification label "{classification}", reply with ONE short sentence that says:
            "This article is related to <category> news."

            Rules:
            - Replace <category> with the label.
            - Do NOT explain the news.
            - Do NOT define the category.
            - Do NOT add extra sentences.
            - Do NOT output anything except the single sentence.

            Examples:
            If label is "Sports" → "This article is related to Sports news."
            If label is "Business" → "This article is related to Business news."
            If label is "World" → "This article is related to World news."
            If label is "Sci/Tech" → "This article is related to Sci/Tech news."
        """

        logger.debug("Calling LLM responder pipeline.")
        out = self.responder_pipeline(prompt, max_length=1000, do_sample=False)[0]["generated_text"]
        # Return the raw generation (strip)
        return out.strip()

if __name__ == "__main__":
    bot = ChatbotInterface()  # defaults: model at models/trained_model.pkl
    
    print("\n🤖 Natural Language Classifier (chatbot mode)")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye 👋")
            break

        try:
            reply = bot.llm_responder(user_input)
        except Exception as e:
            logger.exception("Error during processing")
            reply = "Sorry, something went wrong while processing your request."

        print(f"\nAssistant: {reply}\n")
    
    # bot = ChatbotInterface()

    # result = bot.extract_and_classify("Heiress Nicky Hilton Marries in Vegas, LAS VEGAS - Nicky Hilton, the hotel heiress and socialite, has tied the knot with her beau in a late-night ceremony, according to court filings obtained by The Associated Press.    Hilton, 20, married New York money manager Todd Andrew Meister, 33, at the Las Vegas Wedding Chapel early Sunday, according a Clark County marriage license...")
    # print(result)
    # Heiress Nicky Hilton Marries in Vegas, LAS VEGAS - Nicky Hilton, the hotel heiress and socialite, has tied the knot with her beau in a late-night ceremony, according to court filings obtained by The Associated Press.    Hilton, 20, married New York money manager Todd Andrew Meister, 33, at the Las Vegas Wedding Chapel early Sunday, according a Clark County marriage license...


        # Examples:
        # If label is "Sports" → "This article is related to Sports news."
        # If label is "Business" → "This article is related to Business news."
        # If label is "World" → "This article is related to World news."
        # If label is "Sci/Tech" → "This article is related to Sci/Tech news."