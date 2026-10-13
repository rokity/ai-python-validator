import json
from pathlib import Path

from src.factories.python_correctness import PythonCorrectnessFactory

CONFIG_DIR = Path(__file__).parent / "configs"

MODELS = [
    "bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF:IQ2_XS",  # 0.65 accuracy , failed 56, success 104
    "ggml-org/gemma-3-4b-it-GGUF",  # 0.80 accuracy, failed 32, success 128
    "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",  # 0.76 accuracy, failed 38, success 122
    "Qwen/Qwen2.5-Coder-14B-Instruct-GGUF",  # 0.89 accuracy, failed 18, success 142
]


def load_config(name: str) -> dict:
    with open(CONFIG_DIR / name, encoding="utf-8") as handle:
        return json.load(handle)


def main(model: str = MODELS[0]) -> None:
    exercises = load_config("implementation.json")["items"]
    implementations = [
        {
            "code": impl["code"],
            "label": impl["label"],
            "statement": exercise["statement"],
        }
        for exercise in exercises
        for impl in exercise["implementations"]
    ]

    factory = PythonCorrectnessFactory(model=model)
    report = factory.validate(implementations=implementations)
    print(report.format("Correctness validation results"))


if __name__ == "__main__":
    main()
