from typing import Optional
from transformers import AutoTokenizer
import re

BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"
MIN_TOKENS = 150
MAX_TOKENS = 160
MIN_CHARS = 300
CEILING_CHARS = MAX_TOKENS * 7

class Item:
    """
    Represents a product with its associated price and details.
    """
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    PREFIX = "Price is $"
    QUESTION = "How much does this cost to the nearest dollar?"
    REMOVALS = [
        '"Batteries Included?": "No"', '"Batteries Included?": "Yes"',
        '"Batteries Required?": "No"', '"Batteries Required?": "Yes"',
        "By Manufacturer", "Item", "Date First", "Package", ":", "Number of",
        "Best Sellers", "Number", "Product "
    ]

    def __init__(self, data: dict, price: float):
        self.title = data['title']
        self.price = price
        self.details = None
        self.prompt = None
        self.token_count = 0
        self.include = False
        self.parse(data)

    def scrub_details(self) -> str:
        """Clean the details string by removing common irrelevant text."""
        details = self.details or ""
        for remove in self.REMOVALS:
            details = details.replace(remove, "")
        return details

    def scrub(self, text: str) -> str:
        """
        Remove unnecessary characters and whitespace.
        Also filters out words that are likely irrelevant product numbers.
        """
        text = re.sub(r'[:\[\]"{}【】\s]+', ' ', text).strip()
        text = text.replace(" ,", ",").replace(",,,",",").replace(",,",",")
        words = text.split(' ')
        filtered = [word for word in words if len(word) < 7 or not any(char.isdigit() for char in word)]
        return " ".join(filtered)

    def parse(self, data: dict):
        """Parse the input data and prepare the prompt if within token limits."""
        contents = "\n".join(data.get('description', []))
        if contents:
            contents += "\n"
        features = "\n".join(data.get('features', []))
        if features:
            contents += features + "\n"
        self.details = data.get('details', "")
        if self.details:
            contents += self.scrub_details() + "\n"
        if len(contents) > MIN_CHARS:
            contents = contents[:CEILING_CHARS]
            text = f"{self.scrub(self.title)}\n{self.scrub(contents)}"
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            if len(tokens) > MIN_TOKENS:
                tokens = tokens[:MAX_TOKENS]
                text = self.tokenizer.decode(tokens)
                self.make_prompt(text)
                self.include = True

    def make_prompt(self, text: str):
        """Generate the prompt for training."""
        self.prompt = f"{self.QUESTION}\n\n{text}\n\n{self.PREFIX}{str(round(self.price))}.00"
        self.token_count = len(self.tokenizer.encode(self.prompt, add_special_tokens=False))

    def test_prompt(self) -> str:
        """Return a prompt for testing with the price removed."""
        return self.prompt.split(self.PREFIX)[0] + self.PREFIX

    def __repr__(self) -> str:
        return f"<{self.title} = ${self.price}>"
