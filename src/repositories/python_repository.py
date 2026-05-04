from src.repositories.prompt_repository import PromptRepository


class PythonRepository(PromptRepository):
    def __init__(self, code: str):
        super().__init__("python", code)
