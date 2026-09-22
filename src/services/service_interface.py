from abc import ABC, abstractmethod


class ServiceInterface(ABC):
    def __init__(self, model_name: str, role: str = "user"):
        self.model_name = model_name
        self.role = role

    @abstractmethod
    def chat(self, message: str) -> str:
        pass
