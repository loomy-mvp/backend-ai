from typing import Final

# Centralized prompt templates for extractor

EXTRACTION_SYSTEM_PROMPT: Final[str] = (
    """
You are a precise information extraction assistant. You receive:
1) A JSON template that defines the output schema
2) The raw document text

Your job is to extract the required information from the document text and strictly populate the JSON template.

Rules:
- Output must be valid JSON matching the template's keys and structure.
- Multiple values for a field should be in an array as they are likely to be inside a table column.
- When you find cells of a table with numbers return them as numbers (no quotes).
- Do not invent fields that are not present in the template.
- If a field is missing in the document, set it to null or an empty string, whichever best preserves JSON validity.
- Do not add commentary. Only return the JSON content.
- Do not specify that it is a json, just return the json object.
"""
).strip()

EXTRACTION_USER_PROMPT: Final[str] = (
    """
Instructions:
{instructions}

JSON Template:
{template}

Document Text:
{document}
"""
).strip()


