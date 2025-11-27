import re

def clean_text(text: str):
        
        if not text:
            return ""

        # 1. Strip whitespace
        text = text.strip()

        # 2. Replace multiple spaces/newlines with a single space
        text = re.sub(r"\s+", " ", text)

        # 3. Escape double quotes
        text = text.replace('"', '\\"')

        # 4. Remove control characters
        text = re.sub(r"[\x00-\x1f\x7f]", "", text)

        return text