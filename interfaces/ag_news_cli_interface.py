import logging
from classifier.news_classifier import NewsClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ag_news_cli_interface():
    classifier = NewsClassifier("models/trained_model.pkl")

    print("\nAG News Classifier")
    print("Type 'exit' at any time to quit.\n")
    
    while True:
        title = input("Enter news title: ")
        if title.lower().strip() == "exit":
            break

        description = input("Enter news description: ")
        if description.lower().strip() == "exit":
            break

        prediction = classifier.classify(title, description)

        print(f"\n[Result] {prediction}\n")
        print("-" * 40)

    print("Goodbye")


if __name__ == "__main__":
    cli_interface()
