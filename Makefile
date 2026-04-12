.PHONY: help install dbt-build test predict lint format

help:  ## Show this help message
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n\nTargets:\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

install:  ## Install Python dependencies
	pip install -r requirements.txt

dbt-build:  ## Run dbt pipeline (requires data/raw/ — see Data Notice in README)
	cd transform && dbt run && dbt test

test:  ## Run Task 2 unit tests (8 tests)
	python -m pytest tests/function_tests.py -v

predict:  ## Generate Task 3 predictions (~15 min)
	python src/models/predict_model.py

lint:  ## Check code style with Ruff
	ruff check . && ruff format --check .

format:  ## Auto-fix and format code with Ruff
	ruff check --fix . && ruff format .
