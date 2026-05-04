sample:
	source venv/bin/activate && python -m src.samples.sample


install:
	source venv/bin/activate && pip install -r requirements.txt	


freeze:
	source venv/bin/activate && pip freeze > requirements.txt

validate:
	source venv/bin/activate && ruff check src 

lint:
	source venv/bin/activate && ruff format src/
