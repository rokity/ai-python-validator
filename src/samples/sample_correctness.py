from src.factories.python_correctness import PythonCorrectnessFactory

MODELS = [
    "bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF:IQ2_XS",
    "ggml-org/gemma-3-4b-it-GGUF",
    "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
    "Qwen/Qwen2.5-Coder-14B-Instruct-GGUF",
]
CODE_TO_VALIDATE = [
    "def sum(a, b):\n    return a - b",
    "def sum(a, b):\n    return a + b",
    "def diff(a, b):\n    return a * b",
    "def sum(a, b):\n    return a / b",
]
EXERCISE_DESCRIPTION = "write a python function called 'sum' that returns the sum of two numbers passed as parameters."

if __name__ == "__main__":
    factory = PythonCorrectnessFactory(
        code=CODE_TO_VALIDATE,
        model=MODELS[0],
        exercise=EXERCISE_DESCRIPTION,
        results=[False, True, False, False],
    )
    validation_result = factory.validate()
    print("Final validation result:", validation_result)
