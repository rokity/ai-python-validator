# Generate an abstract class for a prompt repository
from abc import ABC


class PromptRepository(ABC):
    def __init__(self, language: str, code: str):
        self.language = language
        self.code = code

    def get_prompt(self) -> str:
        return "Please validate the syntax of this sample code, just answer yes or no, if it's yes don't say other things just Yes,if it's not valid please be COINCISE explaining where the issue is. The code is ```{language}\n{code}\n```"

    def assemble_prompt(self) -> str:
        prompt = self.get_prompt()
        return prompt.replace("{code}", self.code).replace("{language}", self.language)
