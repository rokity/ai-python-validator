from src.repositories.python_repository import PythonRepository
from src.services.ollama import OllamaService


class DeepSeekPythonFactory:
    def __init__(self, code: str):
        self.code = code
        self.python_prompt = PythonRepository(code).assemble_prompt()

    def validate(self):
        print("Validating Python code with DeepSeek-Coder...")
        ollama_service = OllamaService(model_name="deepseek-coder")
        print("Generated prompt for validation:")
        print(self.python_prompt)
        response = ollama_service.chat(self.python_prompt)
        return response
