from src.repositories.prompt_corretness_repository import PromptCorrectnessRepository
from src.services.llamacpp import LlamaCppService


class PythonCorrectnessFactory:
    def __init__(
        self,
        model: str,
    ):
        self.model = model
        # self.exercise = exercise
        # self.code = code
        # self.results = results
        self.llama_service = LlamaCppService(model_name=self.model)

    def validate(self, implementations: list[dict[str, str, str]]) -> list[str]:
        failed, success = 0, 0
        with self.llama_service.session():
            for implementation_dict in implementations:
                statement = implementation_dict["statement"]
                python_prompt = PromptCorrectnessRepository(
                    language="python",
                    code=implementation_dict["code"],
                    exercise=statement,
                ).assemble_prompt()
                response = self.llama_service.chat(python_prompt)
                response_check = PromptCorrectnessRepository._check_response(response)
                if response_check != implementation_dict["label"]:
                    print("Generated prompt for validation:\n", python_prompt)
                    print(
                        f"Validation mismatch. Expected: {implementation_dict['label']}, Got: {response_check}"
                    )
                    print("LLM response:\n", response)
                    failed += 1
                else:
                    success += 1
        return failed, success
