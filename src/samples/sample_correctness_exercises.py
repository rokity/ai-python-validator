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
    exercises = load_config("exercises.json")["exercises"]
    items = load_config("implementation.json")["items"]
    implementations = [impl for item in items for impl in item["implementations"]]
    code = [impl["code"] for impl in implementations]
    expected = [impl["label"] for impl in implementations]
    exercises = [exercise["statement"] for exercise in exercises]
    # merge exercises and code into a single dictionary where each exercise maps to its implementations
    merged = {}
    for i, exercise in enumerate(exercises):
        merged[exercise] = []
        for j in range(4):
            merged[exercise].append(
                {"code": code[i * 4 + j], "expected": expected[i * 4 + j]}
            )

    factory = PythonCorrectnessFactory(model=model)
    failed, success = factory.validate(exercises=merged)

    print(f"Validation results: failed={failed}, success={success}")
    print(
        f"Accuracy of the model correctness/ total: {success / (failed + success) if (failed + success) > 0 else 0:.2f}"
    )


if __name__ == "__main__":
    main()
