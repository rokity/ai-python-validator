from src.repositories.prompt_repository import PromptRepository
import re


class PromptCorrectnessRepository(PromptRepository):
    def get_prompt(self) -> str:
        return (
            "You're a professor that evaluates the correctness of the following {language} code , this is the exercise '{exercise}' and this is the student implementation : \n"
            "```{language}\n{code}\n```"
            "\nReply with exactly one word: CORRECT or INCORRECT. No explanations, no punctuation, no hints, no reasoning, no results different from the one word response."
            "\nVerify that the student's implementation meets the requirements and the goal of the exercise."
        )

    def assemble_prompt(self) -> str:
        prompt = self.get_prompt()
        return (
            prompt.replace("{code}", self.code)
            .replace("{language}", self.language)
            .replace("{exercise}", self.exercise)
        )

    @staticmethod
    def _check_response(response: str) -> bool | None:
        text = response.lower().strip()
        if re.search(r"\bincorrect\b", text, re.IGNORECASE):
            return False
        if re.search(r"\bcorrect\b", text, re.IGNORECASE):
            return True
        else:
            return None
