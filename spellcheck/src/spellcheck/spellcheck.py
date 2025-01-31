"""Spellcheck module."""
import logging

from spellcheck.model import BaseModel


class Spellcheck:
    """Spellcheck module to correct typos and errors in lists of ingredients.
    
    Args:
        model (BaseModel): model used for correcting list of ingredients.
    """
    def __init__(self, model: BaseModel) -> None:
        self.model = model

    def correct(self, text: str) -> str:
        """Correct list of ingredients
        
        Args:
            list_of_ingredients (str): List of ingredients to correct.

        Return:
            (str): Corrected list of ingredients.
        """
        correction = self.model.generate(text=text)
        return correction
