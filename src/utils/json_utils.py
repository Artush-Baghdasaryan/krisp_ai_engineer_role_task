import json
import re


def extract_json(text: str) -> dict | list:
    """Extract JSON object or array from text. Strips markdown code blocks if present."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)
