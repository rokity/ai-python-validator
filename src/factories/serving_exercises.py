import json
from pathlib import Path

CONFIG_DIR = Path(__file__).parent.parent / "samples" / "configs"


class ServingExercisesFactory:
    def __init__(self):
        pass

    def _load_config(self, name: str) -> list[dict]:
        with open(CONFIG_DIR / name, encoding="utf-8") as handle:
            return json.load(handle)

    def get_exercises(self, course_name: str) -> list[dict]:
        return self._load_config(course_name)

    def get_exercise(self, id: int, course_name: str) -> dict:
        exercises = self._load_config(course_name)
        return next((exercise for exercise in exercises if exercise["id"] == id), None)
