from src.repositories.prompt_syntax_validator_repository import (
    PromptSyntaxValidatorRepository,
)
from src.services.llamacpp import LlamaCppService


class PythonSyntaxValidatorFactory:
    def __init__(self, model: str):
        self.model = model
        self.llama_service = LlamaCppService(model_name=self.model)

    def validate(self, code: list[str], results: list[bool]):
        failed, success = 0, 0
        with self.llama_service.session():
            for implementation in code:
                python_prompt = PromptSyntaxValidatorRepository(
                    language="python", code=implementation
                ).assemble_prompt()
                response = self.llama_service.chat(python_prompt)
                response_check = PromptSyntaxValidatorRepository._check_response(
                    response
                )
                if response_check != results[code.index(implementation)]:
                    print("Generated prompt for validation:\n", python_prompt)
                    print("LLM response:\n", response)
                    print(
                        f"Validation mismatch. Expected: {results[code.index(implementation)]}, Got: {response_check}"
                    )
                    failed += 1
                else:
                    success += 1
        return failed, success
