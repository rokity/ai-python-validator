from ollama import chat
from .service_interface import ServiceInterface


class OllamaService(ServiceInterface):
    def __init__(self, model_name: str, role: str = "user"):
        super().__init__(model_name, role)

    def chat(self, message: str):
        messages = [
            {
                "role": self.role,
                "content": message,
            },
        ]
        response = chat(model=self.model_name, messages=messages)
        return response.message.content
