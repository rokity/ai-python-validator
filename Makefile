sample_syntax_validator:
	python -m src.samples.sample_syntax_validator

sample_correct:
	python -m src.samples.sample_correctness

sample_syntax_exercises:
	python -m src.samples.sample_syntax_exercises

sample_correctness_exercises:
	python -m src.samples.sample_correctness_exercises

sample_complexity_exercises:
	python -m src.samples.sample_complexity_exercises

train_bifi_syntax:
	python -m src.training.bifi_syntax train --data data/bifi/data --dtype float32 --amp $(ARGS)

predict_bifi_syntax:
	python -m src.training.bifi_syntax predict --adapter $(ADAPTER) --code "$(CODE)"

install:
	python -m pip install -r requirements.txt	

freeze:
	pip freeze > requirements.txt

validate:
	ruff check src 

lint:
	ruff format src/

run_frontend:
	cd frontend/my-react-router-app && npm run dev

run_backend:
	cd src/api && python -m fastapi dev

download_datasets:
	mkdir data
	mkdir data/bifi
	curl -fSL -o data_minimal.zip https://nlp.stanford.edu/projects/myasu/BIFI/data_minimal.zip
	unzip data_minimal.zip -d data/bifi
