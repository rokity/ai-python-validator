# Generate an abstract class for a prompt repository
from abc import ABC


class PromptRepository(ABC):
    def __init__(self, language: str, code: str, exercise: str = ""):
        self.language = language
        self.code = code
        self.exercise = exercise

    def get_prompt(self) -> str:
        pass

    def assemble_prompt(self) -> str:
        prompt = self.get_prompt()
        return prompt.replace("{code}", self.code).replace("{language}", self.language)

    @staticmethod
    def _check_response(response: str) -> bool:
        pass
