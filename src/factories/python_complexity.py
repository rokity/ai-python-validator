from src.metrics.classification import ClassificationReport, compute_report
from src.repositories.prompt_complexity_repository import PromptComplexityRepository
from src.services.llamacpp import LlamaCppService


class PythonComplexityFactory:
    def __init__(self, model: str):
        self.model = model
        self.llama_service = LlamaCppService(model_name=self.model)

    def _ask(self, implementation_dict: dict[str, str]) -> str:
        python_prompt = PromptComplexityRepository(
            language="python",
            code=implementation_dict["code"],
            exercise=implementation_dict.get("statement", ""),
        ).assemble_prompt()
        return self.llama_service.chat(python_prompt)

    def analyze_implementation(self, implementations: list[dict[str, str]]) -> str:
        """Return the normalized Big-O class of the first implementation."""
        with self.llama_service.session():
            for implementation_dict in implementations:
                response = self._ask(implementation_dict)
                return PromptComplexityRepository._check_response(response)

    def validate(self, implementations: list[dict[str, str]]) -> ClassificationReport:
        pairs: list[tuple[object, object]] = []
        with self.llama_service.session():
            for implementation_dict in implementations:
                response = self._ask(implementation_dict)
                response_check = PromptComplexityRepository._check_response(response)
                expected = PromptComplexityRepository.normalize(
                    implementation_dict["complexity"]
                )
                if response_check != expected:
                    print(
                        f"Complexity mismatch. Expected: {expected}, Got: {response_check}"
                    )
                    print("LLM response:\n", response)
                pairs.append((expected, response_check))
        return compute_report(pairs)
