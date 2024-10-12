# text_processing.py

import spacy
from typing import List

class TextProcessor:
    """Handle text-related operations such as structure detection and splitting."""

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def detect_document_structure(self, text: str) -> List[str]:
        """Detect logical structure in a document using spaCy."""
        doc = self.nlp(text)
        return [sent.text for sent in doc.sents]

    def split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        return text.split("\n\n")

    def split_by_token_limit(self, text: str, token_limit: int = 512) -> List[str]:
        """Split text by token limit."""
        tokens = text.split()
        return [' '.join(tokens[i:i + token_limit]) for i in range(0, len(tokens), token_limit)]
