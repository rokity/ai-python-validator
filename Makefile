sample_syntax_validator:
	source venv/bin/activate && python -m src.samples.sample_syntax_validator

sample_correct:
	source venv/bin/activate && python -m src.samples.sample_correctness

sample_syntax_exercises:
	source venv/bin/activate && python -m src.samples.sample_syntax_exercises

sample_correctness_exercises:
	source venv/bin/activate && python -m src.samples.sample_correctness_exercises

sample_complexity_exercises:
	source venv/bin/activate && python -m src.samples.sample_complexity_exercises

train_bifi_syntax:
	source venv/bin/activate && python -m src.training.bifi_syntax train --data $(DATA_DIR) $(if $(MODEL),--model $(MODEL))

predict_bifi_syntax:
	source venv/bin/activate && python -m src.training.bifi_syntax predict --adapter $(ADAPTER) --code "$(CODE)" $(if $(MODEL),--model $(MODEL))

install:
	source venv/bin/activate && pip install --ignore-installed --no-deps --dry-run -r requirements.txt	

freeze:
	source venv/bin/activate && pip freeze > requirements.txt

validate:
	source venv/bin/activate && ruff check src 

lint:
	source venv/bin/activate && ruff format src/

run_frontend:
	cd frontend/my-react-router-app && npm run dev

run_backend:
	cd src/api && python -m fastapi dev

download_datasets:
	mkdir data
	mkdir data/bifi
	curl -fSL -o data_minimal.zip https://nlp.stanford.edu/projects/myasu/BIFI/data_minimal.zip
	unzip data_minimal.zip -d data/bifi
