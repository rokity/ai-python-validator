import json
from pathlib import Path

from src.factories.python_complexity import PythonComplexityFactory

CONFIG_DIR = Path(__file__).parent / "configs"

MODELS = [
    "bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF:IQ2_XS",  # Accuracy 0.36
    "ggml-org/gemma-3-4b-it-GGUF",  # Accuracy 0.86
    "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",  # Accuracy 1.00
    "Qwen/Qwen2.5-Coder-14B-Instruct-GGUF",  # Accuracy 0,93
    "unsloth/GLM-4-9B-0414-GGUF:Q4_K_M",  # Accuracy 1.00
    "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF",
    "Qwen/Qwen2.5-Coder-3B-Instruct-GGUF",
]


def load_config(name: str) -> dict:
    with open(CONFIG_DIR / name, encoding="utf-8") as handle:
        return json.load(handle)


def main(model: str = MODELS[4]) -> None:
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
    report = factory.validate(implementations=implementations)
    print(report.format("Complexity validation results"))


if __name__ == "__main__":
    main()
