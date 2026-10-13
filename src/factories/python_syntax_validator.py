from src.metrics.classification import ClassificationReport, compute_report
from src.repositories.prompt_syntax_validator_repository import (
    PromptSyntaxValidatorRepository,
)
from src.services.llamacpp import LlamaCppService


class PythonSyntaxValidatorFactory:
    def __init__(self, model: str):
        self.model = model
        self.llama_service = LlamaCppService(model_name=self.model)

    def validate_syntax(self, code: list[str]):
        with self.llama_service.session():
            for implementation in code:
                python_prompt = PromptSyntaxValidatorRepository(
                    language="python", code=implementation
                ).assemble_prompt()
                response = self.llama_service.chat(python_prompt)
                return response

    def validate(self, code: list[str], results: list[bool]) -> ClassificationReport:
        pairs: list[tuple[object, object]] = []
        with self.llama_service.session():
            for index, implementation in enumerate(code):
                python_prompt = PromptSyntaxValidatorRepository(
                    language="python", code=implementation
                ).assemble_prompt()
                response = self.llama_service.chat(python_prompt)
                response_check = PromptSyntaxValidatorRepository._check_response(
                    response
                )
                expected = results[index]
                if response_check != expected:
                    print("Generated prompt for validation:\n", python_prompt)
                    print("LLM response:\n", response)
                    print(
                        f"Validation mismatch. Expected: {expected}, Got: {response_check}"
                    )
                pairs.append((expected, response_check))
        return compute_report(pairs)
