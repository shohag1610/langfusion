from interfaces.chatbot_interface import ChatbotInterface
from unittest.mock import MagicMock


def test_llm_extractor_parses_valid_json():
    bot = ChatbotInterface()

    bot._extractor_pipeline = MagicMock(return_value=[
        {
            "generated_text": """
            {
                "title": "Symantec Readies Patching Tool",
                "description": "ON IPatch monitors, ensures networked Windows systems are current in midsize businesss."
            }
            """
        }
    ])

    # Call the llm_extractor
    result = bot.llm_extractor(
        'Symantec Readies Patching Tool,"ON IPatch monitors, ensures networked Windows systems are current in midsize businesss.'
    )

    # Assertions
    assert result is not None
    assert isinstance(result, dict)
    assert "title" in result
    assert "description" in result
    assert result["title"] == "Symantec Readies Patching Tool"
    assert result["description"] == "ON IPatch monitors, ensures networked Windows systems are current in midsize businesss."
    
def test_extract_and_classify():
    bot = ChatbotInterface()

    # Mock llm_extractor
    bot.llm_extractor = MagicMock(return_value={
        "title": "Tesla unveils CyberHauler",
        "description": "The vehicle promises 500 miles range and autonomous driving features."
    })

    # Mock classifier
    bot.classifier = MagicMock()
    bot.classifier.classify = MagicMock(return_value="Business")

    user_input = "Some user input about Tesla"
    expected = {
        "title": "Tesla unveils CyberHauler",
        "description": "The vehicle promises 500 miles range and autonomous driving features.",
        "label": "Business"
    }

    result = bot.extract_and_classify(user_input)
    assert result == expected

    # Ensure mocks were called correctly
    bot.llm_extractor.assert_called_once_with(user_input)
    bot.classifier.classify.assert_called_once_with(
        "Tesla unveils CyberHauler",
        "The vehicle promises 500 miles range and autonomous driving features."
    )
