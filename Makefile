.PHONY: setup fmt lint test api

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt

fmt:
	black . && ruff check --fix .

lint:
	ruff check . && black --check .

test:
	pytest -q

api:
	uvicorn api.main:app --reload --port 8000
