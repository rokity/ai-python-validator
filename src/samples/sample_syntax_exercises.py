import ast
import json
from pathlib import Path

from src.factories.python_syntax_validator import PythonSyntaxValidatorFactory

CONFIG_DIR = Path(__file__).parent / "configs"

MODELS = [
    "bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF:IQ2_XS",  # 0.53 accuracy , failed 75, success 85
    "ggml-org/gemma-3-4b-it-GGUF",  # 0.89 accuracy, failed 18, success 142
    "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",  # 0.89 accuracy, falliti 17, success 143
    "Qwen/Qwen2.5-Coder-14B-Instruct-GGUF",  # 0.85, failed 24, success 136
]


def load_config(name: str) -> dict:
    with open(CONFIG_DIR / name, encoding="utf-8") as handle:
        return json.load(handle)


def is_valid_syntax(code: str) -> bool:
    try:
        ast.parse(code)
    except SyntaxError:
        return False
    return True


def main(model: str = MODELS[0]) -> None:
    items = load_config("implementation.json")["items"]
    implementations = [impl for item in items for impl in item["implementations"]]
    code = [impl["code"] for impl in implementations]
    expected = [is_valid_syntax(snippet) for snippet in code]

    factory = PythonSyntaxValidatorFactory(model=model)
    failed, success = factory.validate(code=code, results=expected)

    print(f"Validation results: failed={failed}, success={success}")
    print(
        f"Accuracy of the model correctness/ total: {success / (failed + success) if (failed + success) > 0 else 0:.2f}"
    )


if __name__ == "__main__":
    main()
