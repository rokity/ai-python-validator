from src.factories.serving_exercises import ServingExercisesFactory
from src.factories.python_syntax_validator import PythonSyntaxValidatorFactory
from src.factories.python_correctness import PythonCorrectnessFactory
from src.factories.python_complexity import PythonComplexityFactory
from src.repositories.prompt_complexity_repository import PromptComplexityRepository

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from models.models import RequestExerciseSyntaxValidation

app = FastAPI()

ORIGINS = ["http://localhost", "http://localhost:8080", "http://localhost:5173"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS = [
    "bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF:IQ2_XS",
    "ggml-org/gemma-3-4b-it-GGUF",
    "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
    "Qwen/Qwen2.5-Coder-14B-Instruct-GGUF",
]


@app.get("/exercises_list")
def get_exercises_list():
    return ServingExercisesFactory().get_exercises("beginner_exercises.json")


@app.get("/get_exercise/")
def get_exercise_by_id(id: int):
    return ServingExercisesFactory().get_exercise(id, "beginner_exercises.json")


@app.post("/validate_exercise/")
def validate_exercise(request: RequestExerciseSyntaxValidation):
    factory = PythonSyntaxValidatorFactory(model=MODELS[2])
    validation_result = factory.validate_syntax(code=[request.code])
    if validation_result == "INVALID":
        return {"syntax": "INVALID", "correctness": None, "complexity": None}
    else:
        correctness_factory = PythonCorrectnessFactory(MODELS[2])
        exercise = ServingExercisesFactory().get_exercise(
            int(request.id), "beginner_exercises.json"
        )
        statement = exercise["practice problem"]
        implementations = [
            {
                "code": request.code,
                "statement": statement,
            }
        ]
        result = correctness_factory.validate_implementation(implementations)
        complexity_factory = PythonComplexityFactory(MODELS[2])
        complexity = complexity_factory.analyze_implementation(implementations)
        expected_complexity = PromptComplexityRepository.normalize(
            exercise.get("expected complexity", "")
        )
        return {
            "syntax": "VALID",
            "correctness": result,
            "complexity": complexity,
            "expected_complexity": expected_complexity,
            "complexity_match": (
                None
                if expected_complexity is None or complexity is None
                else complexity == expected_complexity
            ),
        }
