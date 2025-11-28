import re

def extract_ag_news_json(text: str):
    if not text:
        return {}

    start = text.find("{")
    if start == -1:
        return {}

    snippet = text[start:]

    # Match "key": "value..." allowing value to be truncated (no closing quote needed)
    pattern = r'"([^"]+)":\s*"([^"]*)'
    matches = re.findall(pattern, snippet, re.DOTALL)

    result = {key: value.strip() for key, value in matches}

    # Ensure required keys exist
    if "title" not in result:
        result["title"] = text
    if "description" not in result:
        result["description"] = ""

    return result