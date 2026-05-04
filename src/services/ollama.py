from ollama import chat


class OllamaService:
    def __init__(self, model_name: str, role: str = "user"):
        self.model_name = model_name
        self.role = role

    def chat(self, message: str):
        messages = [
            {
                "role": self.role,
                "content": message,
            },
        ]
        response = chat(model=self.model_name, messages=messages)
        return response.message.content
