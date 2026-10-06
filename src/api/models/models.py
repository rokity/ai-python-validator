from pydantic import BaseModel


class RequestExerciseSyntaxValidation(BaseModel):
    id: str
    code: str
