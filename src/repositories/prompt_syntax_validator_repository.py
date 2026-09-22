from src.repositories.prompt_repository import PromptRepository


class PromptSyntaxValidatorRepository(PromptRepository):
    def get_prompt(self) -> str:
        return "Think you are a {language} compiler and you must to validate the code passed in input, verify with you knowledge.Reply with exactly one word: VALID or INVALID. No explanations, no punctuation. The code is ```{language}\n{code}\n```"

    @staticmethod
    def _check_response(response: str) -> bool | None:
        text = response.lower().strip()
        if "invalid" in text:
            return False
        if "valid" in text:
            return True
        return None
