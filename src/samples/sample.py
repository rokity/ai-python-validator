from src.factories.deepseek_python import DeepSeekPythonFactory


if __name__ == "__main__":
    # code_to_validate = """a= 0%5"""
    code_to_validate = 'print("Hello, World!")'
    factory = DeepSeekPythonFactory(
        code=code_to_validate
    )
    validation_result = factory.validate()
    print("Final validation result:", validation_result)
