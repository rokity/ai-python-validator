sample_syntax_validator:
	source venv/bin/activate && python -m src.samples.sample_syntax_validator

sample_correct:
	source venv/bin/activate && python -m src.samples.sample_correctness

sample_syntax_exercises:
	source venv/bin/activate && python -m src.samples.sample_syntax_exercises

sample_correctness_exercises:
	source venv/bin/activate && python -m src.samples.sample_correctness_exercises

install:
	source venv/bin/activate && pip install -r requirements.txt	

freeze:
	source venv/bin/activate && pip freeze > requirements.txt

validate:
	source venv/bin/activate && ruff check src 

lint:
	source venv/bin/activate && ruff format src/
