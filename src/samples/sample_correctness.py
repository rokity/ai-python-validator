from src.factories.python_correctness import PythonCorrectnessFactory

MODELS = [
    "bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF:IQ2_XS",
    "ggml-org/gemma-3-4b-it-GGUF",
    "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
    "Qwen/Qwen2.5-Coder-14B-Instruct-GGUF",
]
EXERCISE_DESCRIPTION = "write a python function called 'sum' that returns the sum of two numbers passed as parameters."
IMPLEMENTATIONS = [
    {
        "code": "def sum(a, b):\n    return a - b",
        "label": False,
        "statement": EXERCISE_DESCRIPTION,
    },
    {
        "code": "def sum(a, b):\n    return a + b",
        "label": True,
        "statement": EXERCISE_DESCRIPTION,
    },
    {
        "code": "def diff(a, b):\n    return a * b",
        "label": False,
        "statement": EXERCISE_DESCRIPTION,
    },
    {
        "code": "def sum(a, b):\n    return a / b",
        "label": False,
        "statement": EXERCISE_DESCRIPTION,
    },
]


if __name__ == "__main__":
    factory = PythonCorrectnessFactory(model=MODELS[0])
    validation_result = factory.validate(
        implementations=IMPLEMENTATIONS,
    )
    print("Final validation result:", validation_result)
