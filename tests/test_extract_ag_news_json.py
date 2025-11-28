import pytest
from utils.extract_ag_new_json import extract_ag_news_json

def test_extract_ag_news_json_llm_output():
    llm_output = """
        Extract a news title and a short description from the user's input. Return a JSON object with keys: title, description. Return NOTHING else.

        User input:
        Symantec Readies Patching Tool,"ON IPatch monitors, ensures networked Windows systems are current in midsize businesss.

        JSON:
        {
            "title": "Symantec Readies Patching Tool",
            "description": "ON IPatch monitors, ensures networked Windows systems are current in midsize businesss."
        }
    """
    expected = {
        "title": "Symantec Readies Patching Tool",
        "description": "ON IPatch monitors, ensures networked Windows systems are current in midsize businesss."
    }

    result = extract_ag_news_json(llm_output)
    assert result == expected