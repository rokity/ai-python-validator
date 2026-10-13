import re

from src.repositories.prompt_repository import PromptRepository

COMPLEXITY_CLASSES = [
    "O(1)",
    "O(log n)",
    "O(n)",
    "O(n log n)",
    "O(n^2)",
    "O(n^3)",
    "O(2^n)",
    "O(n!)",
]

_NORMALIZATION_PATTERNS = [
    (r"o\(\s*n\s*!\s*\)", "O(n!)"),
    (r"o\(\s*2\s*(\^|\*\*)\s*n\s*\)", "O(2^n)"),
    (r"o\(\s*n\s*(\^|\*\*)\s*3\s*\)", "O(n^3)"),
    (r"o\(\s*n\s*(\^|\*\*)\s*2\s*\)", "O(n^2)"),
    (r"o\(\s*n\s*\*?\s*log\s*\(?\s*n\s*\)?\s*\)", "O(n log n)"),
    (r"o\(\s*log\s*\(?\s*n\s*\)?\s*\)", "O(log n)"),
    (r"o\(\s*n\s*\)", "O(n)"),
    (r"o\(\s*1\s*\)", "O(1)"),
]


class PromptComplexityRepository(PromptRepository):
    """Prompt asking the model to classify the time complexity of a snippet."""

    def get_prompt(self) -> str:
        return (
            "You're a professor that evaluates the worst case time complexity of the following {language} code, "
            "this is the exercise '{exercise}' and this is the student implementation : \n"
            "```{language}\n{code}\n```"
            "\nReply with exactly one Big-O class chosen from this list: "
            f"{', '.join(COMPLEXITY_CLASSES)}."
            "\nNo explanations, no punctuation, no reasoning, no results different from the one Big-O class response."
            "\nUse 'n' as the size of the input processed by the implementation."
        )

    def assemble_prompt(self) -> str:
        prompt = self.get_prompt()
        return (
            prompt.replace("{code}", self.code)
            .replace("{language}", self.language)
            .replace("{exercise}", self.exercise)
        )

    @staticmethod
    def _check_response(response: str) -> str:
        """Normalize the model answer into one of COMPLEXITY_CLASSES, or None."""
        if not response:
            return None
        text = response.lower().replace("𝑛", "n")
        for pattern, normalized in _NORMALIZATION_PATTERNS:
            if re.search(pattern, text):
                return normalized
        return None

    @staticmethod
    def normalize(complexity: str) -> str:
        """Normalize an expected complexity label written by a human."""
        return PromptComplexityRepository._check_response(complexity)
