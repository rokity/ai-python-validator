from src.factories.python_syntax_validator import PythonSyntaxValidatorFactory

CODE_TO_VALIDATE = [
    "def x():\n    ",
    "def x():\n    pass",
]

RESULTS = [False, True]

MODELS = [
    "bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF:IQ2_XS",
    "ggml-org/gemma-3-4b-it-GGUF",
    "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
    "Qwen/Qwen2.5-Coder-14B-Instruct-GGUF",
]
if __name__ == "__main__":
    code_to_validate = "def x():\n    "
    factory = PythonSyntaxValidatorFactory(model=MODELS[2])
    validation_result = factory.validate(code=CODE_TO_VALIDATE, results=RESULTS)
    print("Final validation result:", validation_result)
