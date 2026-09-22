import os
from contextlib import contextmanager

from .service_interface import ServiceInterface
from .libs._runtime import chat_session, run_example


class LlamaCppService(ServiceInterface):
    def __init__(self, model_name: str):
        super().__init__(model_name=model_name)
        self._ask = None

    @property
    def _hf_repo(self) -> str:
        return os.getenv("LLAMA_CPP_HF_REPO", self.model_name)

    @contextmanager
    def session(self):
        """Keep one llama-server up for every chat() issued inside the block."""
        with chat_session(
            title=f"llama.cpp {self.model_name} ", hf_repo=self._hf_repo
        ) as ask:
            self._ask = ask
            try:
                yield self
            finally:
                self._ask = None

    def chat(self, message: str) -> str:
        if self._ask is not None:
            return self._ask(message)
        return run_example(
            title=f"llama.cpp {self.model_name} ",
            hf_repo=self._hf_repo,
            system_message="",
            user_message=message,
        )
