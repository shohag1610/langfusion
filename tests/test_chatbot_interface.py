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

def test_llm_responder():
    bot = ChatbotInterface()

    # Mock extract_and_classify output
    bot.extract_and_classify = MagicMock(return_value={
        "title": "Tesla launches new battery",
        "description": "The new model offers longer range and faster charging.",
        "label": "Business"
    })

    # Mock pipeline response
    bot.responder_pipeline = MagicMock(return_value=[
        {"generated_text": "This article is related to Business news."}
    ])

    user_input = "Some input about Tesla"
    result = bot.llm_responder(user_input)

    expected_output = "This article is related to Business news."
    assert result == expected_output

    # Ensure extract_and_classify was called once
    bot.extract_and_classify.assert_called_once_with(user_input)

    bot.responder_pipeline.assert_called_once()