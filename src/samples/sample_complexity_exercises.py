import json
from pathlib import Path

from src.factories.python_complexity import PythonComplexityFactory

CONFIG_DIR = Path(__file__).parent / "configs"

MODELS = [
    "bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF:IQ2_XS", # Accuracy 0.36
    "ggml-org/gemma-3-4b-it-GGUF",  # Accuracy 0.86
    "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF", # Accuracy 1.00
    "Qwen/Qwen2.5-Coder-14B-Instruct-GGUF", #Accuracy 0,93
]


def load_config(name: str) -> dict:
    with open(CONFIG_DIR / name, encoding="utf-8") as handle:
        return json.load(handle)


def main(model: str = MODELS[3]) -> None:
    exercises = load_config("complexity.json")["items"]
    implementations = [
        {
            "code": impl["code"],
            "complexity": impl["complexity"],
            "statement": exercise["statement"],
        }
        for exercise in exercises
        for impl in exercise["implementations"]
    ]

    factory = PythonComplexityFactory(model=model)
    failed, success = factory.validate(implementations=implementations)

    print(f"Validation results: failed={failed}, success={success}")
    print(
        f"Accuracy of the model complexity/ total: {success / (failed + success) if (failed + success) > 0 else 0:.2f}"
    )


if __name__ == "__main__":
    main()
