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

train_oncolab_bifi_syntax:
	python -m src.training.bifi_syntax train --data data/bifi/data --dtype float32 --output /content/drive/MyDrive/bifi-syntax --amp $(ARGS) 

train_onkaggle_introprog_correctness:
	CUDA_VISIBLE_DEVICES=0 python -m src.training.introprog_correctness train --data data/introprog --dtype float32 --amp --output /kaggle/working/introprog-correctness $(ARGS)

predict_bifi_syntax:
	python -m src.training.bifi_syntax predict --adapter $(ADAPTER) --code "$(CODE)"

install:
	python -m pip install -r requirements.txt	

install_colab:
	python -m pip install -q "peft>=0.13" "datasets>=3.0" "accelerate>=1.0" ijson
	python -m pip uninstall -y -q torchao

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
	mkdir -p data
	mkdir -p data/bifi
	curl -fSL -o data_minimal.zip https://nlp.stanford.edu/projects/myasu/BIFI/data_minimal.zip
	unzip data_minimal.zip -d data/bifi

download_introprog:
	mkdir -p data/introprog
	curl -fSL -o data/introprog/dublin_data_train.jsonl https://huggingface.co/datasets/koutch/intro_prog/resolve/main/data/dublin_data_train.jsonl
	curl -fSL -o data/introprog/dublin_data_test.jsonl https://huggingface.co/datasets/koutch/intro_prog/resolve/main/data/dublin_data_test.jsonl
