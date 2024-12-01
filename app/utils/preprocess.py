# app/utils/preprocess.py

import re

def clean_text(text: str) -> str:
    """
    Cleans the input text by removing special characters and converting to lowercase.
    """
    text = text.lower()
    text = re.sub(r"[^a-záéíóúñü¿?¡!0-9\s]", "", text)
    return text
