import re

URL_PATTERN = re.compile(r"http\S+|www\S+")
MULTISPACE = re.compile(r"\s+")

def clean_text(text: str) -> str:

    text = str(text).lower()
    text = URL_PATTERN.sub(" ", text)
    text = MULTISPACE.sub(" ", text)

    return text.strip()